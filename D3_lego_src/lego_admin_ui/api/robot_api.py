import logging

from flask import Blueprint, Response, jsonify
from sqlalchemy import func as sa_func

from database.connection import Session
from database.models import Task, RobotLog, RobotAction, LogStatus, Inventory
from state.robot_state import robot_state, state_lock

log = logging.getLogger(__name__)

robot_bp = Blueprint("robot", __name__)


@robot_bp.route("/api/state")
def api_state() -> Response:
    """
    현재 로봇 상태 스냅샷을 반환한다.

    처리 흐름:
    1. state_lock 획득 후 robot_state 복사
    2. 복사본을 JSON 직렬화 (락 구간 최소화)

    Returns:
        joints: {j1~j6} (deg 단위),
        tcp: {x,y,z,rx,ry,rz} (mm / deg 단위, robot base 좌표계),
        status: 현재 동작 설명 문자열,
        ros_connected: ROS 연결 여부,
        safety_mode: 안전 모드 활성 여부
    """
    with state_lock:
        return jsonify({
            "joints":             dict(robot_state["joints"]),
            "tcp":                dict(robot_state["tcp"]),
            "status":             robot_state["status"],
            "ros_connected":      robot_state["ros_connected"],
            "safety_mode":        robot_state["safety_mode"],
            "automation_running": robot_state["automation_running"],
            "robot_hw_state":     robot_state["robot_hw_state"],
        })


@robot_bp.route("/api/stats")
def api_stats() -> Response:
    """
    픽·플레이스 성공률과 진행 중인 작업 진척도를 반환한다.

    처리 흐름:
    1. RobotLog 집계로 pick/place 성공률 계산
    2. 진행 중인 Task(end_at=None)의 brick_path 기반 진척도 계산
    3. 집계 오류 시 0값 응답 (서버 재시작 없이 복구 가능)

    Returns:
        ok, pick_success_rate, place_success_rate (%), progress (%),
        progress_done, progress_total

    Note:
        총 pick 횟수가 0이면 성공률을 0.0으로 처리해 ZeroDivisionError 방지
    """
    db = Session()
    try:
        total_picks   = db.query(RobotLog).filter(RobotLog.robot_action == RobotAction.pick).count()
        total_success = db.query(RobotLog).filter(RobotLog.robot_action == RobotAction.pick,
                                                   RobotLog.status == LogStatus.success).count()
        # 픽 횟수가 0이면 나눗셈 불가 → 0.0 반환
        pick_success_rate: float = round(total_success / total_picks * 100, 1) if total_picks > 0 else 0.0

        total_places   = db.query(RobotLog).filter(RobotLog.robot_action == RobotAction.place).count()
        total_place_ok = db.query(RobotLog).filter(RobotLog.robot_action == RobotAction.place,
                                                   RobotLog.status == LogStatus.success).count()
        place_success_rate: float = round(total_place_ok / total_places * 100, 1) if total_places > 0 else 0.0

        progress: float = 0.0
        progress_done: int = 0
        progress_total: int = 0
        running: Task | None = (
            db.query(Task)
            .filter(Task.customer_id.isnot(None), Task.brick_path.isnot(None), Task.end_at.is_(None))
            .order_by(Task.created_at.asc())
            .first()
        )
        if running and running.brick_path:
            bricks         = running.brick_path.get("bricks", [])
            progress_total = len(bricks)
            # place success의 MAX(step_order) + 1 = 실제 배치 완료 블럭 수
            # (단순 count()는 한 step당 move/pick/place 3개 로그가 쌓여서 부풀려짐)
            last_done = (
                db.query(sa_func.max(RobotLog.step_order))
                .filter(
                    RobotLog.task_id      == running.id,
                    RobotLog.robot_action == RobotAction.place,
                    RobotLog.status       == LogStatus.success,
                )
                .scalar()
            )
            progress_done = (last_done + 1) if last_done is not None else 0
            if progress_total > 0:
                progress = round(progress_done / progress_total * 100, 1)

        return jsonify({
            "ok": True,
            "pick_success_rate":  pick_success_rate,
            "place_success_rate": place_success_rate,
            "progress":           progress,
            "progress_done":      progress_done,
            "progress_total":     progress_total,
        })
    except Exception:
        log.exception("[STATS] 통계 조회 실패")
        return jsonify({"ok": False, "pick_success_rate": 0, "place_success_rate": 0,
                        "progress": 0, "progress_done": 0, "progress_total": 0})
    finally:
        db.close()


@robot_bp.route("/api/inventory")
def api_inventory() -> Response:
    """
    현재 블럭 재고(빨강/파랑/노랑 × 2x2/2x3)를 반환한다.

    Returns:
        ok, red_2x2, red_2x3, blue_2x2, blue_2x3, yellow_2x2, yellow_2x3, total_blocks
    """
    db = Session()
    try:
        inv: Inventory | None = db.query(Inventory).first()
        if inv is None:
            return jsonify({"ok": True, "red_2x2": 0, "red_2x3": 0,
                            "blue_2x2": 0, "blue_2x3": 0,
                            "yellow_2x2": 0, "yellow_2x3": 0, "total_blocks": 0})
        return jsonify({
            "ok": True,
            "red_2x2":      inv.red_2x2,
            "red_2x3":      inv.red_2x3,
            "blue_2x2":     inv.blue_2x2,
            "blue_2x3":     inv.blue_2x3,
            "yellow_2x2":   inv.yellow_2x2,
            "yellow_2x3":   inv.yellow_2x3,
            "total_blocks": inv.total_blocks,
        })
    except Exception:
        log.exception("[INV] 재고 조회 실패")
        return jsonify({"ok": False})
    finally:
        db.close()


@robot_bp.route("/api/safety/toggle", methods=["POST"])
def safety_toggle() -> Response:
    """
    안전 모드를 토글하고 변경된 상태를 반환한다.

    Returns:
        ok=True, safety_mode: 변경 후 상태(bool)
    """
    with state_lock:
        robot_state["safety_mode"] = not robot_state["safety_mode"]
        mode: bool = robot_state["safety_mode"]
    log.info("[SAFETY] 안전 모드 %s", "ON" if mode else "OFF")
    return jsonify({"ok": True, "safety_mode": mode})
