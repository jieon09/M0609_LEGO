import logging
import os
import subprocess
import threading
import time

from config import ROS_SOURCE, SCRIPT_DIR
from state.robot_state import robot_state, state_lock
from process.process_manager import pick_proc_manager

log = logging.getLogger(__name__)


def do_stop_sync(timeout: int = 10) -> bool:
    """
    [자동화 전용] 로봇 정지 ROS 노드를 동기 실행한다.

    처리 흐름:
    1. subprocess로 robot_base_motion stop 노드 기동
    2. timeout 내 종료 대기
    3. returncode 0이면 True 반환

    Args:
        timeout: 프로세스 최대 대기 시간 (초). 기본 10초.

    Returns:
        True: 정상 종료 (returncode=0)
        False: timeout 또는 비정상 종료

    Note:
        blocking 호출 — 호출 스레드가 timeout 동안 점유됨.
        stop 노드는 서비스 1회 호출 후 즉시 종료하므로 timeout은 여유롭게 설정.
    """
    env = os.environ.copy()
    env.setdefault("DISPLAY", ":0")

    proc = subprocess.Popen(
        ["bash", "-c", f"{ROS_SOURCE} && ros2 run robot_base_motion stop"],
        cwd=SCRIPT_DIR,
        env=env,
    )
    log.info("[STOP_SYNC] PID=%d 시작", proc.pid)

    try:
        proc.wait(timeout=timeout)
        log.info("[STOP_SYNC] PID=%d 종료 returncode=%d", proc.pid, proc.returncode)
        return proc.returncode == 0
    except subprocess.TimeoutExpired:
        proc.kill()
        log.error("[STOP_SYNC] PID=%d timeout(%ds) → 강제 종료", proc.pid, timeout)
        return False


def _monitor_stop(proc: subprocess.Popen) -> None:
    """
    stop 프로세스 종료를 감시하고 robot_state를 갱신하는 내부 함수.

    처리 흐름:
    1. proc.wait()로 종료 대기 (blocking)
    2. 종료 후 robot_state를 "정지 완료"로 갱신
    3. 2초 후 "대기 중"으로 복귀 (상태가 바뀌지 않았을 때만)

    Note:
        daemon 스레드에서 실행 — 메인 프로세스 종료 시 자동 회수됨.
    """
    proc.wait()
    log.info("[STOP] 프로세스 종료 PID=%d returncode=%d", proc.pid, proc.returncode)
    with state_lock:
        robot_state["status"] = "정지 완료"
        robot_state["action"] = None
    time.sleep(2)
    with state_lock:
        if robot_state["status"] == "정지 완료":
            robot_state["status"] = "대기 중"


def do_stop_async() -> None:
    """
    [버튼 전용] 실행 중인 pick/place 프로세스를 종료하고 로봇 정지 노드를 비동기로 기동한다.

    /api/action/stop 버튼에서 호출. API가 즉시 응답해야 하므로 비동기 실행.
    성공/실패는 robot_state["status"]로 확인.

    처리 흐름:
    1. pick_proc_manager.kill_all()로 실행 중인 프로세스 강제 종료
    2. robot_base_motion stop 노드 실행
    3. pick_proc_manager에 등록 (kill_all 대상에 포함)
    4. 종료 감시 스레드(daemon) 기동

    Note:
        kill_all은 비동기 pick/place 프로세스만 대상으로 한다.
        stop 노드는 /dsr01/motion/move_stop 서비스에 1회 요청 후 즉시 종료한다.
    """
    env = os.environ.copy()
    env.setdefault("DISPLAY", ":0")

    pick_proc_manager.kill_all()
    log.info("[STOP] 실행 중 프로세스 강제 종료 완료")

    with state_lock:
        robot_state["status"] = "정지 중..."

    proc = subprocess.Popen(
        ["bash", "-c", f"{ROS_SOURCE} && ros2 run robot_base_motion stop"],
        cwd=SCRIPT_DIR,
        env=env,
    )
    log.info("[STOP] 프로세스 시작 PID=%d", proc.pid)
    pick_proc_manager.add(proc)
    threading.Thread(target=_monitor_stop, args=(proc,), daemon=True).start()
