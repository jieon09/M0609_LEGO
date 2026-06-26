import logging
import uuid

from database.connection import Session
from database.models import RobotLog, Inventory, RobotAction, LogStatus, ErrorCode
from state.robot_state import robot_state, state_lock
from ros.cam_state import cam_state, cam_lock

log = logging.getLogger(__name__)


def write_step_log(
    db: Session,
    task_id: uuid.UUID,
    step_order: int | None,
    robot_action: RobotAction,
    status: LogStatus | None = None,
    error_code: ErrorCode | None = None,
) -> None:
    """
    현재 관절각·TCP 스냅샷과 함께 로봇 동작 로그를 DB에 기록한다.

    처리 흐름:
    1. state_lock 구간에서 joints·tcp 복사 (최소 락 시간)
    2. RobotLog 엔티티 생성 후 commit

    Args:
        db: 열린 SQLAlchemy 세션
        task_id: 로그를 귀속시킬 Task UUID
        step_order: 시퀀스 내 단계 번호 (0-based)
        robot_action: 동작 종류 (running/pick/place)
        status: 처리 결과 상태 (success/failed 등). 생략 시 NULL
        error_code: 실패 원인 코드. 생략 시 NULL

    Note:
        joints 단위: deg (robot base 좌표계)
        tcp 단위: x,y,z=mm / rx,ry,rz=deg (robot base 좌표계)
    """
    with state_lock:
        js:  dict[str, float] = dict(robot_state["joints"])
        tcp: dict[str, float] = dict(robot_state["tcp"])

    entry = RobotLog(
        task_id=task_id,
        robot_action=robot_action,
        step_order=step_order,
        status=status,
        error_code=error_code,
        j1=js["j1"], j2=js["j2"], j3=js["j3"],
        j4=js["j4"], j5=js["j5"], j6=js["j6"],
        tcp_x=tcp["x"],   tcp_y=tcp["y"],   tcp_z=tcp["z"],
        tcp_rx=tcp["rx"], tcp_ry=tcp["ry"], tcp_rz=tcp["rz"],
    )
    db.add(entry)
    db.commit()
    log.info("[AUTO] robot_log 기록 step=%d action=%s status=%s", step_order, robot_action.value, status)


def update_inventory_from_detections(db: Session) -> int:
    """
    YOLO 탐지 결과를 집계해 Inventory 테이블을 갱신한다.

    처리 흐름:
    1. cam_state["detections"] 스냅샷
    2. color+shape 키로 카운트 집계
    3. Inventory 단일 행 업데이트 후 commit

    Args:
        db: 열린 SQLAlchemy 세션

    Returns:
        갱신된 total_blocks 값 (0이면 탐지 결과 없음)

    Note:
        Inventory 테이블은 싱글톤(1행)이라 first()로 바로 접근.
        탐지 결과 없으면 전 항목을 0으로 초기화해 이전 값이 남지 않도록 함.
    """
    with cam_lock:
        detections: list[dict] = list(cam_state["detections"])

    counts: dict[str, int] = {
        "red_2x2": 0, "red_2x3": 0,
        "blue_2x2": 0, "blue_2x3": 0,
        "yellow_2x2": 0, "yellow_2x3": 0,
    }

    if not detections:
        log.warning("[AUTO] YOLO 탐지 결과 없음 → inventory 전체 0으로 초기화")
    else:
        for det in detections:
            key = f"{det.get('color', '').lower()}_{det.get('shape', '')}"
            if key in counts:
                counts[key] += 1

    inv: Inventory | None = db.query(Inventory).first()
    if inv is None:
        log.warning("[AUTO] inventory 행 없음 → 업데이트 건너뜀")
        return 0

    inv.red_2x2      = counts["red_2x2"]
    inv.red_2x3      = counts["red_2x3"]
    inv.blue_2x2     = counts["blue_2x2"]
    inv.blue_2x3     = counts["blue_2x3"]
    inv.yellow_2x2   = counts["yellow_2x2"]
    inv.yellow_2x3   = counts["yellow_2x3"]
    inv.total_blocks = sum(counts.values())
    db.commit()
    log.info("[AUTO] inventory 업데이트 완료 %s total=%d", counts, inv.total_blocks)
    return inv.total_blocks
