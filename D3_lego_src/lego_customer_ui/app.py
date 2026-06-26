import json
import glob
import logging
import os
import sys
import time
import subprocess
import threading
from datetime import datetime
from typing import Optional, Callable, Generator

from dotenv import load_dotenv
load_dotenv()

import cv2
from flask import Flask, render_template, Response, jsonify, request, send_from_directory

from camera import ImageProcessor, SharedState, CameraManager
from camera.config import SAVE_DIR
from database.connection import Session
from database.models import Task, Inventory, RobotLog, RobotAction, LogStatus
from sqlalchemy import func as sa_func

log = logging.getLogger(__name__)

app = Flask(__name__)

_state = SharedState()
_proc = ImageProcessor()
_camera = CameraManager(_state, _proc)

_SEG_SCRIPT  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'services', 'object_segment_fin.py')
_CONV_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'services', 'lego_convert_fin.py')


def _stream(get_fn: Callable[[], Optional[bytes]]) -> Generator[bytes, None, None]:
    _camera.ensure_running()
    while True:
        data = get_fn()
        if data is not None:
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n'
                + data +
                b'\r\n'
            )
        time.sleep(0.033)  # ~30 fps 상한 — sleep 없으면 빈 프레임 구간에서 CPU 100% 도달
        # while True 탈출 조건 없음 — 클라이언트 연결 해제 시 Flask가 generator를 GC해서 자동 종료됨


@app.route('/')
def index() -> str:
    return render_template('index.html')


@app.route('/capture', methods=['POST'])
def capture_image() -> Response:
    # ensure_running: 카메라가 꺼진 상태에서도 캡처 버튼을 누르면 자동 재시작.
    # reset_explicit_stop: 이전 주문 등록 후 explicit stop 상태였다면 해제.
    _camera.reset_explicit_stop()
    _camera.ensure_running()
    filename, label = _camera.capture_manual()
    return jsonify({'ok': bool(filename), 'filename': filename, 'message': label})


@app.route('/detection_status')
def detection_status() -> Response:
    data = _state.get_detection()
    data['pending_draw_file'] = _state.get_pending_draw()
    return jsonify(data)


@app.route('/captures/<path:filename>')
def serve_capture(filename: str) -> Response:
    return send_from_directory(SAVE_DIR, filename)


@app.route('/start_draw', methods=['POST'])
def start_draw() -> Response:
    data = request.get_json(silent=True) or {}
    customer_id_str = str(data.get('customer_id', '')).strip()

    if not customer_id_str:
        return jsonify({'ok': False, 'message': '고객 ID가 필요합니다'})

    capture = _state.get_pending_capture()
    if capture is None:
        return jsonify({'ok': False, 'message': '캡처된 이미지가 없습니다'})

    raw, contour = capture

    # [:30]: 긴 이름이 OS 파일명 255자 제한에 걸리지 않도록 앞부분만 사용
    # 슬래시 제거: customer_id에 경로 구분자가 들어오면 SAVE_DIR 밖으로 파일이 저장될 수 있음
    safe_prefix = customer_id_str[:30].replace('/', '_').replace('\\', '_')
    if contour is not None:
        # contour가 있으면 종이 윤곽 기준으로 원근 보정 후 저장
        filename = _proc.save_warped(raw, contour, prefix=safe_prefix)
    else:
        # contour 없음 = 종이 미감지 → 원본 프레임 그대로 저장
        filename = datetime.now().strftime(f'{safe_prefix}_%Y%m%d_%H%M%S.png')
        cv2.imwrite(os.path.join(SAVE_DIR, filename), raw)

    img_path = os.path.join(SAVE_DIR, filename)

    db = Session()
    try:
        task = Task(customer_id=customer_id_str, img_path=img_path)
        db.add(task)
        db.commit()
        # db.close() 이전에 str로 변환 — 세션 닫힌 후 task.id에 접근하면 DetachedInstanceError 발생
        task_id = str(task.id)
    except Exception as e:
        db.rollback()
        return jsonify({'ok': False, 'message': f'DB 저장 실패: {e}'})
    finally:
        db.close()

    _state.clear_pending_capture()
    # 파이프라인은 저장된 파일을 사용하므로 라이브 피드 불필요 — 리소스 해제.
    # explicit=True: 뒤늦은 video_feed 스트림이 ensure_running을 호출해도 재시작 안 됨.
    _camera.stop(explicit=True)

    def _run_pipeline() -> None:
        # img_path, task_id는 문자열이므로 락 없이 클로저 캡처해서 다른 스레드에서 읽어도 안전
        img_dir      = os.path.dirname(os.path.abspath(img_path))
        segments_dir = os.path.join(img_dir, 'segments')

        log.info('[pipeline] object_segment 시작: %s', img_path)
        # subprocess.run: 외부 스크립트 완료까지 blocking. timeout=120초 — GPU 없는 환경 대비
        seg = subprocess.run(
            [sys.executable, _SEG_SCRIPT,
             '--image',  img_path,
             '--output', img_dir,
             '--no-paper',
             '--no-show'],
            capture_output=True, text=True, timeout=120
        )
        if seg.returncode != 0:
            log.error('[pipeline] object_segment 실패:\n%s', seg.stderr)
            return
        log.info('[pipeline] object_segment 완료')

        log.info('[pipeline] lego_convert 배치 시작: %s', segments_dir)
        # timeout=120초 — 이미지 수가 많을 경우 변환 시간이 길어질 수 있음
        conv = subprocess.run(
            [sys.executable, _CONV_SCRIPT,
             '--batch',   segments_dir,
             '--no-show'],
            capture_output=True, text=True, timeout=120
        )
        if conv.returncode != 0:
            log.error('[pipeline] lego_convert 실패:\n%s', conv.stderr)
            return
        log.info('[pipeline] lego_convert 완료')

        obj_jpg_files = sorted(glob.glob(os.path.join(segments_dir, 'obj*.jpg')))
        if not obj_jpg_files:
            log.warning('[pipeline] obj*.jpg 파일 없음')
            return

        if len(obj_jpg_files) > 1:
            log.info('[pipeline] %d개 객체 발견 → 웹 선택 대기', len(obj_jpg_files))
            _state.set_pending_obj_selection(obj_jpg_files)
            # 최대 300초 blocking — 고객이 선택을 완료할 때까지 파이프라인 중단
            result = _state.wait_obj_selection(timeout=300)
            _state.clear_pending_obj_selection()
            # result=None은 300초 타임아웃 — 고객 무응답 시 첫 번째 객체로 자동 진행
            chosen_idx = result[0] if result else 0
        else:
            chosen_idx = 0

        chosen_jpg  = obj_jpg_files[chosen_idx]
        base_name   = os.path.splitext(os.path.basename(chosen_jpg))[0]
        chosen_json = os.path.join(segments_dir, f'{base_name}_bricks.json')

        if not os.path.exists(chosen_json):
            log.error('[pipeline] 선택된 JSON 없음: %s', chosen_json)
            return

        with open(chosen_json, 'r', encoding='utf-8') as f:
            brick_data = json.load(f)

        # 바깥 db 세션은 이미 닫혔으므로 파이프라인 전용 세션을 새로 열어야 함
        _db = Session()
        try:
            task_obj = _db.query(Task).filter(Task.id == task_id).first()
            if task_obj:
                task_obj.brick_path = brick_data
                task_obj.img_path   = chosen_jpg
                _db.commit()
                log.info('[pipeline] DB 저장 완료 (brick_path 설정): %s', os.path.basename(chosen_jpg))
        except Exception as e:
            _db.rollback()
            log.error('[pipeline] DB 저장 실패: %s', e)
        finally:
            _db.close()

    # daemon=True: Flask 서버 종료 시 파이프라인 스레드도 자동 정리
    threading.Thread(target=_run_pipeline, daemon=True).start()

    return jsonify({
        'ok': True,
        'task_id': task_id,
        'filename': filename,
        'message': f'주문 등록 완료 (작업 ID: {task_id})'
    })


@app.route('/inventory')
def inventory() -> Response:
    # 레고 재고 현황 반환 — 관리자 UI가 DB를 직접 업데이트하면 여기서 반영됨
    db = Session()
    try:
        row = db.query(Inventory).first()
        if row is None:
            return jsonify({'ok': True, 'inventory': {}})
        result = {
            'red_2x2':    row.red_2x2,
            'red_2x3':    row.red_2x3,
            'blue_2x2':   row.blue_2x2,
            'blue_2x3':   row.blue_2x3,
            'yellow_2x2': row.yellow_2x2,
            'yellow_2x3': row.yellow_2x3,
        }
        return jsonify({'ok': True, 'inventory': result})
    except Exception as e:
        return jsonify({'ok': False, 'inventory': {}, 'message': str(e)})
    finally:
        db.close()


@app.route('/task_status')
def task_status() -> Response:
    db = Session()
    try:
        tasks = (
            db.query(Task)
            .order_by(Task.created_at.desc())
            .limit(15)
            .all()
        )
        result = [
            {
                'id':          str(t.id)[:8].upper(),  # UUID 앞 8자만 표시용으로 사용 — 조회 키로 쓰지 않음
                'customer_id': t.customer_id or '—',
                'status':      'done' if t.end_at else ('ready' if t.brick_path else 'processing'),
                'created_at':  t.created_at.strftime('%m/%d %H:%M:%S') if t.created_at else '—',
            }
            for t in tasks
        ]
        return jsonify({'ok': True, 'tasks': result})
    except Exception as e:
        return jsonify({'ok': False, 'tasks': [], 'message': str(e)})
    finally:
        db.close()


@app.route('/build_status/<task_id>')
def build_status(task_id: str) -> Response:
    db = Session()
    try:
        task = db.query(Task).filter(Task.id == task_id).first()
        if not task:
            return jsonify({'ok': False, 'message': '작업 없음'})

        bricks = []
        if isinstance(task.brick_path, dict):
            bricks = task.brick_path.get('bricks', [])

        # place success 상태의 step_order만 집계 — running/failed/재시도 중인 step은 제외.
        # MAX +1 = 배치 완료 브릭 수 (step_order는 0부터 시작).
        last_done = db.query(sa_func.max(RobotLog.step_order)) \
            .filter(RobotLog.task_id == task_id) \
            .filter(RobotLog.robot_action == RobotAction.place) \
            .filter(RobotLog.status == LogStatus.success) \
            .scalar()
        placed_count = (last_done + 1) if last_done is not None else 0

        return jsonify({
            'ok':           True,
            'bricks':       bricks,
            'placed_count': placed_count,
            'total_count':  len(bricks),
            'is_done':      task.end_at is not None,
        })
    except Exception as e:
        return jsonify({'ok': False, 'message': str(e)})
    finally:
        db.close()


@app.route('/object_selection_status')
def object_selection_status() -> Response:
    # 폴링용 — 프론트가 주기적으로 호출해 객체 선택 UI 표시 여부를 판단
    files = _state.get_pending_obj_selection()
    if files is None:
        return jsonify({'pending': False})
    return jsonify({'pending': True, 'count': len(files)})


@app.route('/object_preview/<int:idx>')
def object_preview(idx: int) -> Response:
    files = _state.get_pending_obj_selection()
    if files is None or idx >= len(files):
        return jsonify({'error': 'not found'}), 404
    path = files[idx]
    return send_from_directory(os.path.dirname(path), os.path.basename(path))


@app.route('/confirm_object_selection', methods=['POST'])
def confirm_object_selection() -> Response:
    data     = request.get_json(silent=True) or {}
    selected = data.get('selected_indices', [])
    _state.confirm_obj_selection(selected)
    return jsonify({'ok': True})


@app.route('/dismiss_draw', methods=['POST'])
def dismiss_draw() -> Response:
    # 그리기 취소 — 대기 중인 캡처/그림 상태를 초기화하고 카메라 라이브 피드 재개.
    # reset_explicit_stop: 이전 주문 등록 후 explicit stop 상태였다면 해제해야
    # ensure_running이 실제로 재시작한다.
    _state.clear_pending_draw()
    _state.clear_pending_capture()
    _camera.reset_explicit_stop()
    _camera.ensure_running()
    return jsonify({'ok': True})


@app.route('/video_feed')
def video_feed() -> Response:
    return Response(
        _stream(_state.get_frame),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/video_feed_hsv')
def video_feed_hsv() -> Response:
    # 색상 튜닝 시 디버깅 목적으로만 사용
    def _get_hsv() -> Optional[bytes]:
        raw = _state.get_raw()
        if raw is None:
            return None
        ok, buf = cv2.imencode('.jpg', cv2.cvtColor(raw, cv2.COLOR_BGR2HSV))
        return buf.tobytes() if ok else None

    return Response(
        _stream(_get_hsv),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/video_feed_debug')
def video_feed_debug() -> Response:
    return Response(
        _stream(_state.get_debug),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


if __name__ == '__main__':
    import socket
    host_ip = socket.gethostbyname(socket.gethostname())
    print(f"\n * 고객 UI 실행 중")
    print(f" * Local:   http://127.0.0.1:5000/")
    print(f" * Network: http://{host_ip}:5000/\n")
    app.run(debug=True)
