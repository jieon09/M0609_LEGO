import logging
from typing import Any

from flask import Blueprint, Response, jsonify, request

from state.robot_state import robot_state, state_lock
from process.process_manager import pick_proc_manager
from services.pick_service import do_pick_async
from services.place_service import do_place_async, do_place_one_async
from services.go_home_service import do_go_home_async
from services.stop_service import do_stop_async
from services.robot_recovery_service import do_recovery_async

log = logging.getLogger(__name__)

action_bp = Blueprint("action", __name__)


@action_bp.route("/api/action/pick", methods=["POST"])
def action_pick() -> Response:
    """
    색상·형태 필터를 적용해 블럭 집기 동작을 비동기 시작한다.

    처리 흐름:
    1. 진행 중인 pick 프로세스 강제 종료
    2. robot_state 갱신 (UI 즉시 반영)
    3. 백그라운드 스레드로 pick 파이프라인 위임

    Returns:
        ok=True JSON — 실제 성공 여부는 robot_state.status로 추적
    """
    body: dict[str, Any] = request.get_json(silent=True) or {}
    filter_color: str = body.get("color", "all")
    filter_shape: str = body.get("shape", "all")
    log.info("[API] POST /api/action/pick color=%s shape=%s", filter_color, filter_shape)

    # 이전 pick 작업이 남아 있으면 충돌 방지를 위해 먼저 종료
    pick_proc_manager.kill_all()
    with state_lock:
        robot_state["status"] = "블럭 탐지 중…"
        robot_state["action"] = "pick"

    do_pick_async(filter_color, filter_shape)
    return jsonify({"ok": True, "message": "블럭 집기를 시작합니다. (YOLO 자동 켜기 → 탐지 → 끄기 → pick)"})


@action_bp.route("/api/action/place", methods=["POST"])
def action_place() -> Response:
    """
    마지막으로 집은 블럭을 기본 위치에 놓는 동작을 비동기 시작한다.

    처리 흐름:
    1. 진행 중인 프로세스 강제 종료
    2. robot_state 갱신
    3. 백그라운드 스레드로 place 파이프라인 위임

    Returns:
        ok=True JSON
    """
    log.info("[API] POST /api/action/place 요청")
    # 이전 작업과 충돌 방지
    pick_proc_manager.kill_all()
    with state_lock:
        robot_state["status"] = "블럭 놓기 수행 중…"
        robot_state["action"] = "place"

    do_place_async()
    return jsonify({"ok": True, "message": "블럭 놓기 동작을 시작했습니다."})


@action_bp.route("/api/action/stop", methods=["POST"])
def action_stop() -> Response:
    """
    실행 중인 모든 동작을 즉시 취소하고 로봇을 정지시킨다.

    처리 흐름:
    1. 실행 중인 pick/place 프로세스 강제 종료
    2. automation_cancel 플래그 설정 (자동화 루프 중단 신호)
    3. 로봇 정지 노드 비동기 실행

    Returns:
        ok=True JSON
    """
    log.info("[API] POST /api/action/stop 요청")
    pick_proc_manager.kill_all()
    with state_lock:
        robot_state["automation_cancel"] = True
    do_stop_async()
    return jsonify({"ok": True, "message": "취소 요청을 전송했습니다."})


@action_bp.route("/api/action/go_home", methods=["POST"])
def action_go_home() -> Response:
    """
    실행 중인 프로세스를 중단하고 로봇을 홈 위치로 복귀시킨다.

    처리 흐름:
    1. 진행 중인 pick/place 프로세스 강제 종료
    2. go_home 노드를 비동기 실행

    Returns:
        ok=True JSON
    """
    log.info("[API] POST /api/action/go_home 요청")
    pick_proc_manager.kill_all()
    do_go_home_async()
    return jsonify({"ok": True, "message": "홈 위치로 이동을 시작합니다."})


@action_bp.route("/api/action/place_one", methods=["POST"])
def action_place_one() -> Response:
    """
    지정한 그리드 셀에 특정 크기의 블럭 하나를 놓는 동작을 비동기 시작한다.

    처리 흐름:
    1. row, col, width, height 파라미터 필수 검증
    2. 진행 중인 프로세스 강제 종료
    3. robot_state 갱신
    4. 백그라운드 스레드로 place_one 파이프라인 위임

    Args (JSON body):
        row: 배치 행 인덱스 (0-based)
        col: 배치 열 인덱스 (0-based)
        width: 블럭 너비 (스터드 단위)
        height: 블럭 높이 (스터드 단위)

    Returns:
        ok=True JSON, 파라미터 누락 시 400 오류
    """
    body: dict[str, Any] = request.get_json(silent=True) or {}
    row:    Any = body.get("row")
    col:    Any = body.get("col")
    width:  Any = body.get("width")
    height: Any = body.get("height")

    if any(v is None for v in (row, col, width, height)):
        return jsonify({"ok": False, "message": "row, col, width, height 모두 필요합니다."}), 400

    log.info("[API] POST /api/action/place_one row=%s col=%s width=%s height=%s",
             row, col, width, height)

    pick_proc_manager.kill_all()
    with state_lock:
        robot_state["status"] = "블럭 놓기 수행 중…"
        robot_state["action"] = "place"

    do_place_one_async(row, col, width, height)
    return jsonify({"ok": True, "message": f"row={row} col={col} {width}x{height} 블럭 놓기를 시작합니다."})


@action_bp.route("/api/action/recovery", methods=["POST"])
def action_recovery() -> Response:
    """안전 정지(State 5) 복구 명령을 비동기로 실행한다."""
    log.info("[API] POST /api/action/recovery 요청")
    do_recovery_async()
    return jsonify({"ok": True, "message": "복구 명령을 전송했습니다."})
