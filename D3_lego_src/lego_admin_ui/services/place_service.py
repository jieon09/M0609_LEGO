import logging
import os
import subprocess
import threading
import time

from config import ROS_SOURCE, BLOCK_PLACE_DIR as BLOCK_PICKER_DIR
from state.robot_state import robot_state, state_lock
from process.process_manager import pick_proc_manager

log = logging.getLogger(__name__)

def do_place_sync(row: int, col: int, width: int, height: int, timeout: int = 60) -> bool:
    """
    [자동화 전용] 지정 그리드 셀에 블럭을 놓는 ROS 노드를 동기 실행한다.

    automation_service의 pick → place 순서 보장을 위해 blocking으로 실행.
    성공/실패 여부를 반환값으로 받아 다음 스텝 진행 여부를 결정한다.

    처리 흐름:
    1. ROS 파라미터로 row/col/width/height 전달
    2. subprocess blocking 대기 (최대 timeout초)
    3. returncode 0이면 True 반환

    Args:
        row: 배치 행 인덱스 (0-based)
        col: 배치 열 인덱스 (0-based)
        width: 블럭 너비 (스터드 단위)
        height: 블럭 높이 (스터드 단위)
        timeout: 프로세스 최대 대기 시간 (초). 기본 60초.

    Returns:
        True: 정상 종료 (returncode=0)
        False: timeout 또는 비정상 종료

    Note:
        blocking 호출.
        60초는 가장 먼 배치 셀까지의 이동+놓기 실측 상한값 기준.
    """
    env = os.environ.copy()
    env.setdefault("DISPLAY", ":0")

    ros_args = f"--ros-args -p row:={row} -p col:={col} -p width:={width} -p height:={height}"
    proc = subprocess.Popen(
        ["bash", "-c", f"{ROS_SOURCE} && ros2 run place_block place_block16 {ros_args}"],
        cwd=BLOCK_PICKER_DIR,
        env=env,
    )
    pick_proc_manager.add(proc)
    log.info("[PLACE_SYNC] PID=%d 시작 row=%d col=%d %dx%d", proc.pid, row, col, width, height)

    try:
        proc.wait(timeout=timeout)
        log.info("[PLACE_SYNC] PID=%d 종료 returncode=%d", proc.pid, proc.returncode)
        return proc.returncode == 0
    except subprocess.TimeoutExpired:
        proc.kill()
        log.error("[PLACE_SYNC] PID=%d timeout(%ds) → 강제 종료", proc.pid, timeout)
        return False


def _monitor_place(proc: subprocess.Popen) -> None:
    """
    place 프로세스 종료를 감시하고 robot_state를 갱신하는 내부 함수.

    처리 흐름:
    1. proc.wait()로 종료 대기 (blocking)
    2. 종료 후 robot_state를 "블럭 놓기 완료"로 갱신
    3. 2초 후 "대기 중"으로 복귀 (상태가 바뀌지 않았을 때만)

    Note:
        daemon 스레드에서 실행 — 메인 프로세스 종료 시 자동 회수됨.
        2초 지연은 UI가 완료 메시지를 사용자에게 보여주기 위한 최소 표시 시간.
    """
    proc.wait()
    log.info("[PLACE] 프로세스 종료 PID=%d returncode=%d", proc.pid, proc.returncode)
    with state_lock:
        robot_state["status"] = "블럭 놓기 완료"
        robot_state["action"] = None
    time.sleep(2)
    with state_lock:
        if robot_state["status"] == "블럭 놓기 완료":
            robot_state["status"] = "대기 중"


def do_place_async() -> None:
    """
    [버튼 전용] 파라미터 없이 place 노드를 비동기로 기동한다.

    /api/action/place 버튼에서 호출. API가 즉시 응답해야 하므로 비동기 실행.
    노드 내부 기본값으로 동작하며, 성공/실패는 robot_state["status"]로 확인.

    처리 흐름:
    1. place_block16 노드를 파라미터 없이 실행
    2. pick_proc_manager에 등록 (kill_all 대상에 포함)
    3. 종료 감시 스레드(daemon) 기동

    Note:
        ROS 파라미터를 생략하면 노드 내부 기본값으로 동작한다.
    """
    env = os.environ.copy()
    env.setdefault("DISPLAY", ":0")

    proc = subprocess.Popen(
        ["bash", "-c", f"{ROS_SOURCE} && ros2 run place_block place_block16"],
        cwd=BLOCK_PICKER_DIR,
        env=env,
    )
    log.info("[PLACE] 프로세스 시작 PID=%d", proc.pid)
    pick_proc_manager.add(proc)
    # daemon=True: 메인 프로세스 종료 시 감시 스레드 자동 회수
    threading.Thread(target=_monitor_place, args=(proc,), daemon=True).start()


def do_place_one_async(row: int, col: int, width: int, height: int) -> None:
    """
    [버튼 전용] 지정 그리드 셀에 블럭 하나를 놓는 place 노드를 비동기로 기동한다.

    /api/action/place_one 버튼에서 호출. API가 즉시 응답해야 하므로 비동기 실행.
    성공/실패는 robot_state["status"]로 확인.

    처리 흐름:
    1. row/col/width/height를 ROS 파라미터로 전달해 place_block16 실행
    2. pick_proc_manager에 등록
    3. 종료 감시 스레드(daemon) 기동

    Args:
        row: 배치 행 인덱스 (0-based)
        col: 배치 열 인덱스 (0-based)
        width: 블럭 너비 (스터드 단위)
        height: 블럭 높이 (스터드 단위)
    """
    env = os.environ.copy()
    env.setdefault("DISPLAY", ":0")

    ros_args = f"--ros-args -p row:={row} -p col:={col} -p width:={width} -p height:={height}"
    proc = subprocess.Popen(
        ["bash", "-c", f"{ROS_SOURCE} && ros2 run place_block place_block16 {ros_args}"],
        cwd=BLOCK_PICKER_DIR,
        env=env,
    )
    log.info("[PLACE_ONE] 프로세스 시작 PID=%d", proc.pid)
    pick_proc_manager.add(proc)
    threading.Thread(target=_monitor_place, args=(proc,), daemon=True).start()
