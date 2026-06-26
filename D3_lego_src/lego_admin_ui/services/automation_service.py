import logging
import threading
import time
import uuid
from datetime import datetime, timezone

from database.connection import Session
from database.models import Task, RobotAction, RobotLog
from state.robot_state import robot_state, state_lock
from ros.cam_state import cam_state, cam_lock
from ros.hand_state import hand_state, hand_lock
from ros.voice_state import voice_state, voice_lock
from services.pick_service import do_pick_sync, do_move_pick_place_sync
from process.process_manager import pick_proc_manager
from services.place_service import do_place_sync
from services.go_home_service import do_go_home_sync
from services.stop_service import do_stop_sync
from services.robot_log_service import write_step_log, update_inventory_from_detections
from database.models import LogStatus, ErrorCode
from voice.voice_tts import speak

# 로깅 객체 생성
log = logging.getLogger(__name__)

# [설정] DB 및 주문서에서 사용하는 색상 단축키를 ROS/YOLO API용 전체 영문 이름으로 매핑
_COLOR_MAP: dict[str, str] = {"r": "red", "b": "blue", "y": "yellow"}

def _automation_loop() -> None:
    """
    주문 큐(DB)를 지속적으로 폴링(Polling)하여 'Pick → Place' 사이클을 완전 자동 실행하는 메인 루프.

    전체 처리 흐름:
    1. [대기] 2초 간격으로 아직 처리되지 않은 Task(주문)를 DB에서 FIFO(선입선출) 방식으로 조회
    2. [검증] 해당 주문의 brick_path(블럭 경로/배치도) 데이터의 정방성 및 유효성 검사
    3. [이동] 작업 시작 전, 로봇을 Pick 및 Place가 가능한 최적의 '준비 위치'로 이동
    4. [비전] 비전(YOLO) 카메라를 일시 활성화하여 작업 공간 내의 블럭 재고 상태를 실시간 갱신
    5. [루프] step_order 순서에 맞게 안전 센서(손 감지)를 체크하며 개별 블럭 Pick & Place 반복 수행
    6. [종료] 모든 작업을 마치면 로봇을 홈 위치(Safe Home)로 복귀시키고 비전 센서 종료 및 상태 갱신

    비동기 처리 주의사항 (Concurrency Note):
        - 본 루프는 별도의 상태머신(FSM) 없이 단일 while-True 구조로 순차 실행(Synchronous)됩니다.
        - 따라서 공유 자원(robot_state, cam_state, hand_state) 접근 시 반드시 전용 스레드 락(Lock)을 획득해야 합니다.
        - 한 번에 오직 하나의 Task만 처리하며 동시 작업은 지원하지 않습니다.
    """
    log.info("[AUTO] 자동화 루프 시작")

    while True:
        # [폴링 주기 설정] 2초 대기 — DB 호출 부하를 줄이면서 사용자의 주문에 실시간 대응하기 위한 최적값
        time.sleep(2)
        
        # 각 폴링 세션마다 독립된 DB 세션을 생성하여 커넥션 풀 고갈 및 세션 오염 방지
        db = Session()
        try:
            # ------------------------------------------------------------------
            # 1. 대기 중인 주문 탐지 (FIFO: 가장 오래된 미완료 작업 우선 처리)
            # ------------------------------------------------------------------
            customer: Task | None = (
                db.query(Task)
                .filter(
                    Task.customer_id.isnot(None),  # 고객 ID가 존재하고
                    Task.brick_path.isnot(None),    # 배치 경로 데이터가 있으며
                    Task.end_at.is_(None),          # 아직 완료되지 않은 작업
                )
                .order_by(Task.created_at.asc())    # 생성시간 기준 오름차순 (First-In, First-Out)
                .first()
            )

            if customer is None:
                log.info("[AUTO] 등록된 주문이 없습니다")
                continue

            customer_id: uuid.UUID = customer.id
            log.info("[AUTO] 주문 발견 id=%s customer_id=%s", customer_id, customer.customer_id)

            # ------------------------------------------------------------------
            # 2. brick_path 데이터 유효성 검증
            # ------------------------------------------------------------------
            brick_path_data: dict | None = customer.brick_path
            if not brick_path_data or "bricks" not in brick_path_data:
                log.warning("[AUTO] brick_path 데이터 포맷이 올바르지 않습니다. (bricks 키 누락)")
                continue

            bricks: list[dict] = brick_path_data["bricks"]
            if not bricks:
                # 배열이 비어있다면 비정상적인 주문이므로 DB에서 삭제하여 무한 루프 방지
                log.warning("[AUTO] bricks 배열이 비어있음 → 비정상 Task 데이터 삭제 처리")
                db.delete(customer)
                db.commit()
                continue

            log.info("[AUTO] 작업 시작, 총 %d 스텝 수행 예정", len(bricks))

            # 전역 로봇 상태 안전하게 변경 (웹/App 모니터링 연동용)
            with state_lock:
                robot_state["status"]             = "자동화 실행 중"
                robot_state["automation_running"] = True

            # ------------------------------------------------------------------
            # 3. step_order 기반의 하드웨어 제어 시퀀스 시작 (준비이동 → YOLO 감지 → Pick → Place 반복)
            # ------------------------------------------------------------------
            success: bool   = True

            # 이전에 성공한 마지막 place 스텝 다음부터 재개 (취소/재시작 시 진행도 보존)
            last_done = (
                db.query(RobotLog)
                  .filter(RobotLog.task_id == customer_id,
                          RobotLog.robot_action == RobotAction.place,
                          RobotLog.status == LogStatus.success)
                  .order_by(RobotLog.step_order.desc())
                  .first()
            )
            step_order: int = (last_done.step_order + 1) if last_done else 0
            if step_order > 0:
                log.info("[AUTO] 이전 진행 이력 발견 → step_order=%d 부터 재개", step_order)

            while True:
                # [취소 요청 체크] 외부에서 automation_cancel=True 설정 시 즉시 중단
                with state_lock:
                    _cancel = robot_state["automation_cancel"]
                    if _cancel:
                        robot_state["automation_cancel"] = False
                        robot_state["status"] = "자동화 취소됨"
                if _cancel:
                    log.warning("[AUTO] 취소 요청 감지 → 자동화 중단")
                    success = False
                    break

                # [안전 인터록] 손 감지 또는 음성 일시정지 시 즉시 대기
                while True:
                    with hand_lock:
                        _hand = hand_state["detected"]
                    with voice_lock:
                        _voice = voice_state["paused"]
                    if not _hand and not _voice:
                        break

                    log.warning("[AUTO] 손 감지 또는 음성 일시정지 — 해제 대기 중")
                    with state_lock:
                        robot_state["status"] = "긴급 정지: 손 감지됨" if _hand else "음성 명령: 일시 정지됨"
                    time.sleep(1)

                # [종료 조건] 현재 Task의 모든 블럭 배치를 성공적으로 완료한 경우
                if step_order >= len(bricks):
                    log.info("[AUTO] step_order=%d → 모든 스텝이 완벽히 완료되었습니다.", step_order)
                    
                    # Task 완료 시간 기록 (end_at 자체로 완료 표시)
                    customer.end_at = datetime.now(timezone.utc)
                    db.commit()
                    log.info("[AUTO] Task end_at 최종 기록 완료 id=%s", customer_id)
                    
                    # 로봇 안전 홈 위치 복귀 시퀀스
                    with state_lock:
                        robot_state["status"] = "홈 위치로 이동 중..."
                    do_go_home_sync()
                    write_step_log(db, customer_id, step_order, RobotAction.go_home, status=LogStatus.success)
                    break

                # --------------------------------------------------------------
                # [데이터 파싱] 현재 단계(step_order)에서 다룰 블럭 정보 추출 및 매핑
                # --------------------------------------------------------------
                brick: dict = bricks[step_order]
                _c: str     = str(brick.get("color", "all")).lower()
                color: str  = _COLOR_MAP.get(_c, _c)  # 단축어('r')인 경우 전체이름('red')으로 치환
                width: int  = int(brick.get("width",  1))
                height: int = int(brick.get("height", 1))
                row: int    = int(brick.get("row",    0))
                col: int    = int(brick.get("col",    0))

                # 블럭의 가로x세로 크기를 조합하여 로봇 그리퍼 알고리즘용 형태(Shape) 정의
                wh: tuple[int, int] = (width, height)
                if wh == (2, 2):
                    shape: str = "2x2"
                elif wh in ((2, 3), (3, 2)):
                    shape = "2x3"
                else:
                    shape = "all"

                log.info("[AUTO] step_order=%d → Pick 대상(%s, 형태:%s) / Place 목적지(행:%d, 열:%d, 크기:%dx%d)",
                         step_order, color, shape, row, col, width, height)

                # --------------------------------------------------------------
                # [동작 1] Pick 준비 위치로 로봇 암 이동 (실패 시 최대 1회 재시도 법칙)
                # --------------------------------------------------------------
                with state_lock:
                    robot_state["status"] = f"step {step_order}: pick 준비 위치로 이동 중"
                log.info("[AUTO] step_order=%d pick 준비 위치로 이동 개시", step_order)

                move_ok = False
                for attempt in range(2):
                    # 이동 시작 로그 기록 (Running 상태)
                    write_step_log(db, customer_id, step_order, RobotAction.move_pick_place, status=LogStatus.running)
                    
                    if do_move_pick_place_sync():
                        move_ok = True
                        break  # 이동 성공 시 재시도 루프 즉시 탈출
                        
                    log.warning("[AUTO] pick 준비 위치 이동 실패 (시도 횟수=%d/2)", attempt + 1)
                    write_step_log(db, customer_id, step_order, RobotAction.move_pick_place, status=LogStatus.failed)
                    
                    if attempt == 0:
                        log.info("[AUTO] 기구학적 물리 물리 불안정성 해소를 위해 2초 대기 후 재시도합니다.")
                        time.sleep(2)

                # 이동 최종 실패 시 처리 로직
                if not move_ok:
                    with hand_lock:
                        _hand = hand_state["detected"]
                    # 만약 실패 원인이 중간에 개입한 '손 감지 인터록' 때문이라면, 스텝을 폭파하지 않고 처음부터 재시도
                    if _hand:
                        log.warning("[AUTO] step_order=%d 이동 실패 원인이 '손 감지'로 확인됨 → 손 해제 후 처음부터 다시 시도", step_order)
                        continue
                    
                    # 손 감지가 아님에도 이동 실패 시(하드웨어 에러, 기구학적 기이점 등) 전체 자동화 중단
                    log.error("[AUTO] step_order=%d pick 준비 위치 최종 이동 실패 → 전체 공정 락아웃(중단)", step_order)
                    with state_lock:
                        robot_state["status"] = "자동화 실패: 준비 이동 실패"
                    success = False
                    break

                # 준비 위치 이동 성공 기록 및 하드웨어 잔여 진동 흡수를 위한 2초 정지 타임아웃
                write_step_log(db, customer_id, step_order, RobotAction.move_pick_place, status=LogStatus.success)
                log.info("[AUTO] step_order=%d pick 준비 위치 이동 완료, 물리 진동 감쇠를 위해 2초 정지", step_order)
                time.sleep(2)

                # --------------------------------------------------------------
                # [동작 2] 비전 센서(YOLO) 활성화 및 실시간 블럭 감지 시퀀스
                # --------------------------------------------------------------
                with cam_lock:
                    cam_state["yolo_on"]    = True   # 비전 쓰레드에게 카메라 캡처 및 추론 활성화 명령
                    cam_state["detections"] = []     # 이전 단계의 감지 잔재 버퍼 초기화
                with state_lock:
                    robot_state["status"] = f"step {step_order}: 블럭 감지 중..."
                log.info("[AUTO] step_order=%d YOLO 비전 모델 활성화, 물체 인식 대기", step_order)

                # 카메라 프레임에 블럭이 하나 이상 포착될 때까지 1초 간격으로 대기 (폴링 블로킹)
                while True:
                    time.sleep(1)
                    with cam_lock:
                        _current = cam_state["detections"]
                    if len(_current) > 0:
                        log.info("[AUTO] 카메라 프레임 내 블럭 감지 완료 (%d개 인식)", len(_current))
                        break
                    log.debug("[AUTO] 블럭이 카메라에 포착되지 않음. 대기 중...")

                # 인식이 완료되었으므로 카메라 자원 및 연산량 절약을 위해 YOLO 비활성화
                with cam_lock:
                    cam_state["yolo_on"] = False

                # [재고 반영] 카메라가 탐지한 실시간 데이터를 기반으로 DB 재고 테이블(Inventory) 업데이트
                # Note: 탐지 데이터 정보(_current)는 다음 프로세스인 do_pick_sync() 내부에서 조준용으로 소비하므로 초기화하지 않음
                total_detected: int = update_inventory_from_detections(db)

                # 카메라 화면에 잡힌 자재가 아무것도 없는 경우 (자재 고갈 상황)
                if total_detected == 0:
                    log.warning("[AUTO] step_order=%d 재고 갱신 결과 0개 발견 → 자재 부족으로 공정 긴급 중단", step_order)
                    speak("블럭이 없습니다! 블럭을 채워주세요.")
                    
                    # 더 이상의 위험 움직임을 방지하기 위해 로봇 구동 즉시 중단(정지 명령 패킷 송신)
                    do_stop_sync()
                    time.sleep(3)
                    
                    with state_lock:
                        robot_state["status"] = "블럭 없음 (자동화 실패)"
                    success = False
                    break

                # --------------------------------------------------------------
                # [동작 3] PICK - 손 감지 시 즉시 정지, 해제 후 pick 재개
                # --------------------------------------------------------------
                # pick 재시도 루프: 손 감지 인터럽트면 pick만 재시도, 실제 pick 실패면 YOLO 재감지로 분기
                pick_ok = False
                while True:
                    with state_lock:
                        robot_state["status"] = f"step {step_order}: 블럭 집기 중"
                    write_step_log(db, customer_id, step_order, RobotAction.pick, status=LogStatus.running)

                    # do_pick_sync는 블로킹 subprocess이므로 별도 스레드에서 실행하고
                    # 메인 스레드에서 0.2초 간격으로 손 감지를 폴링하여 즉시 중단 가능하게 함
                    pick_result: list[bool] = [False]
                    pick_done = threading.Event()

                    def _run_pick(result=pick_result, done=pick_done) -> None:
                        result[0] = do_pick_sync(color, shape)
                        done.set()

                    threading.Thread(target=_run_pick, daemon=True).start()

                    pick_interrupted = False
                    while not pick_done.wait(timeout=0.2):
                        with hand_lock:
                            if hand_state["detected"]:
                                pick_interrupted = True
                                break
                        with voice_lock:
                            if voice_state["paused"]:
                                pick_interrupted = True
                                break

                    if pick_interrupted:
                        # 손 감지 또는 음성 정지 → pick 프로세스 즉시 강제 종료 후 로봇 정지
                        # 인터럽트 원인을 확인해서 error_code를 분리 기록 — 손이 우선순위 (동시 발생 시 손이 더 즉각적 안전 이슈)
                        with hand_lock:
                            _hand_now = hand_state["detected"]
                        _pick_err = ErrorCode.HAND_DETECTED if _hand_now else ErrorCode.VOICE_PAUSED
                        pick_proc_manager.kill_all()
                        pick_done.wait()
                        do_stop_sync()
                        log.warning("[AUTO] step_order=%d Pick 중 인터럽트(%s) → 로봇 정지, 해제 후 pick 재개",
                                    step_order, _pick_err.value)
                        write_step_log(db, customer_id, step_order, RobotAction.pick,
                                       status=LogStatus.failed, error_code=_pick_err)
                        while True:
                            with hand_lock:
                                _hand = hand_state["detected"]
                            with voice_lock:
                                _voice = voice_state["paused"]
                            if not _hand and not _voice:
                                break
                            with state_lock:
                                robot_state["status"] = "긴급 정지: 손 감지됨 (pick 대기)" if _hand else "음성 명령: 일시 정지됨 (pick 대기)"
                            time.sleep(1)
                        log.info("[AUTO] step_order=%d 인터럽트 해제 확인 → pick 재시도", step_order)
                        continue  # pick 재시도 루프 처음으로

                    if pick_result[0]:
                        pick_ok = True
                        break  # pick 성공 → place로

                    # 손 없이 pick 실패 (그리퍼 미끄러짐, 조준 실패 등) → 자재 재인식 필요
                    log.warning("[AUTO] step_order=%d Pick 동작 실패 → 자재 재인식을 위해 스텝 처음으로 복귀", step_order)
                    write_step_log(db, customer_id, step_order, RobotAction.pick, status=LogStatus.failed)
                    break  # pick 루프 탈출 → 아래 if not pick_ok에서 스텝 전체 재시도

                if not pick_ok:
                    continue  # 메인 루프로 (move + YOLO 포함 전체 스텝 재시도)

                # Pick 성공 시 그리퍼 압력 안정화 및 파지 확인을 위해 1초 대기
                log.info("[AUTO] step_order=%d Pick 동작 성공, 1초 그리퍼 안정화 대기", step_order)
                write_step_log(db, customer_id, step_order, RobotAction.pick, status=LogStatus.success)
                time.sleep(1)

                # --------------------------------------------------------------
                # [동작 4] PLACE (지정된 좌표에 블럭 내려놓기 및 탈착)
                # --------------------------------------------------------------
                # 그리퍼에 블럭이 물려있으므로 손 감지 시 pick 재시도 없이 place만 반복한다.
                place_ok = False
                while True:
                    with state_lock:
                        robot_state["status"] = f"step {step_order}: 블럭 놓기 중"
                    write_step_log(db, customer_id, step_order, RobotAction.place, status=LogStatus.running)

                    # do_place_sync는 블로킹 subprocess이므로 스레드에서 실행하고
                    # 메인 스레드에서 0.2초 간격으로 손 감지를 폴링하여 즉시 중단 가능하게 함
                    place_result: list[bool] = [False]
                    place_done = threading.Event()

                    def _run_place(result=place_result, done=place_done) -> None:
                        result[0] = do_place_sync(row, col, width, height)
                        done.set()

                    threading.Thread(target=_run_place, daemon=True).start()

                    place_interrupted = False
                    while not place_done.wait(timeout=0.2):
                        with hand_lock:
                            if hand_state["detected"]:
                                place_interrupted = True
                                break
                        with voice_lock:
                            if voice_state["paused"]:
                                place_interrupted = True
                                break

                    if place_interrupted:
                        # 손 감지 또는 음성 정지 → place 프로세스 즉시 강제 종료 후 로봇 정지
                        # 인터럽트 원인을 확인해서 error_code를 분리 기록 — 손이 우선순위
                        with hand_lock:
                            _hand_now = hand_state["detected"]
                        _place_err = ErrorCode.HAND_DETECTED if _hand_now else ErrorCode.VOICE_PAUSED
                        pick_proc_manager.kill_all()
                        place_done.wait()
                        do_stop_sync()
                        log.warning("[AUTO] step_order=%d Place 중 인터럽트(%s) → 로봇 정지, 해제 대기",
                                    step_order, _place_err.value)
                        write_step_log(db, customer_id, step_order, RobotAction.place,
                                       status=LogStatus.failed, error_code=_place_err)
                        while True:
                            with hand_lock:
                                _hand = hand_state["detected"]
                            with voice_lock:
                                _voice = voice_state["paused"]
                            if not _hand and not _voice:
                                break
                            with state_lock:
                                robot_state["status"] = "긴급 정지: 손 감지됨 (place 대기)" if _hand else "음성 명령: 일시 정지됨 (place 대기)"
                            time.sleep(1)
                        log.info("[AUTO] step_order=%d 인터럽트 해제 확인 → place 재시도", step_order)
                        continue  # place 재시도 루프 처음으로 (pick은 건너뜀)

                    if place_result[0]:
                        place_ok = True
                        break  # Place 성공 시 루프 탈출

                    # 손 없이 place 실패 → 기구학/통신 에러로 조립 공정 정지
                    write_step_log(db, customer_id, step_order, RobotAction.place, status=LogStatus.failed)
                    log.error("[AUTO] step_order=%d Place 작업 최종 실패 → 하드웨어 안전 잠금 및 작업 중단", step_order)
                    break

                # Place 최종 실패 시, 바깥쪽 메인 제어 루프를 깨뜨리고 전체 Task 실패 처리
                if not place_ok:
                    success = False
                    break

                # 현재 스텝의 블럭 배치 성공 시 다음 블럭으로 포인터 이동
                log.info("[AUTO] step_order=%d Place 완료 성공", step_order)
                write_step_log(db, customer_id, step_order, RobotAction.place, status=LogStatus.success)
                step_order += 1

            # --------------------------------------------------------------
            # 4. 자원 정리 및 최종 시퀀스 종료 처리 (Finally 가기 전 마무리)
            # --------------------------------------------------------------
            # 혹시 모를 비전 카메라 리소스의 잔여 활성화를 막기 위한 안전 셧다운
            with cam_lock:
                cam_state["yolo_on"]    = False
                cam_state["detections"] = []

            # 성공 유무에 따른 최종 상태 문자열 확정 및 시스템 상태 반영
            final_status: str = "자동화 완료" if success else "자동화 실패"
            with state_lock:
                robot_state["status"]             = final_status
                robot_state["automation_running"] = False
            log.info("[AUTO] 단일 주문 프로세스 최종 종료 id=%s → 결과: %s", customer_id, final_status)

        except Exception:
            # 예기치 못한 크리티컬 시스템 에러(DB 단절, 널 포인트 참조 등) 발생 시 롤백 및 로깅
            log.exception("[AUTO] 자동화 루프 수행 중 시스템 예외(Exception) 발생")
            with state_lock:
                robot_state["automation_running"] = False
            try:
                db.rollback()
            except Exception:
                pass
        finally:
            # 트랜잭션 종료 후 세션을 명시적으로 닫아 DB 커넥션 누수(Leak) 방지
            db.close()


def start_automation_loop() -> None:
    """
    백그라운드에서 상시 구동될 수 있도록 자동화 메인 루프를 Daemon 스레드로 기동합니다.

    Note:
        - daemon=True 속성을 통해 메인 웹 서버나 메인 프로세스가 종료될 경우, 
          하위 로봇 제어 스레드도 좀비 프로세스가 되지 않고 즉시 안전하게 동반 종료됩니다.
    """
    threading.Thread(target=_automation_loop, daemon=True).start()