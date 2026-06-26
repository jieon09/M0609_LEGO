import logging
import os
import subprocess
import threading
import time

from config import ROS_SOURCE, SCRIPT_DIR
from state.robot_state import robot_state, state_lock
from process.process_manager import pick_proc_manager

log = logging.getLogger(__name__)


def do_recovery_sync(timeout: int = 30) -> bool:
    """
    [자동화 전용] 안전 정지(Safe Stop) 복구 노드를 동기 실행한다.

    안전 정지(노란색) 복구 흐름:
        1. drl_script_stop  — 실행 중인 DRL 스크립트 정지
        2. call_set_robot_control(2)  — 리셋 명령 전송
        3. State 5(Safe Stop) → State 1(STANDBY) 즉시 복구
           서보가 꺼지지 않으므로 별도 서보온 불필요

    Args:
        timeout: 프로세스 최대 대기 시간 (초). 기본 30초.

    Returns:
        True: 정상 종료 (returncode=0)
        False: timeout 또는 비정상 종료
    """
    env = os.environ.copy()
    env.setdefault("DISPLAY", ":0")

    proc = subprocess.Popen(
        ["bash", "-c", f"{ROS_SOURCE} && ros2 run cobot1 check_robot_state_recovery"],
        cwd=SCRIPT_DIR,
        env=env,
    )
    pick_proc_manager.add(proc)
    log.info("[RECOVERY_SYNC] PID=%d 시작", proc.pid)

    try:
        proc.wait(timeout=timeout)
        log.info("[RECOVERY_SYNC] PID=%d 종료 returncode=%d", proc.pid, proc.returncode)
        return proc.returncode == 0
    except subprocess.TimeoutExpired:
        proc.kill()
        log.error("[RECOVERY_SYNC] PID=%d timeout(%ds) → 강제 종료", proc.pid, timeout)
        return False


def _monitor_recovery(proc: subprocess.Popen) -> None:
    proc.wait()
    log.info("[RECOVERY] 프로세스 종료 PID=%d returncode=%d", proc.pid, proc.returncode)
    if proc.returncode != 0:
        with state_lock:
            robot_state["status"] = "대기 중"
            robot_state["action"] = None
        return
    with state_lock:
        robot_state["status"] = "복구 완료"
        robot_state["action"] = None
    time.sleep(2)
    with state_lock:
        if robot_state["status"] == "복구 완료":
            robot_state["status"] = "대기 중"


def do_recovery_async() -> None:
    """
    [버튼 전용] 안전 정지(Safe Stop) 복구 노드를 비동기로 기동한다.

    안전 정지(노란색) 복구 흐름:
        1. drl_script_stop  — 실행 중인 DRL 스크립트 정지
        2. call_set_robot_control(2)  — 리셋 명령 전송
        3. State 5(Safe Stop) → State 1(STANDBY) 즉시 복구

    처리 흐름:
        1. 실행 중인 pick/place 프로세스 강제 종료
        2. check_robot_state_recovery.py 노드 실행
        3. pick_proc_manager에 등록
        4. 종료 감시 스레드(daemon) 기동
    """
    env = os.environ.copy()
    env.setdefault("DISPLAY", ":0")

    pick_proc_manager.kill_all()

    with state_lock:
        robot_state["status"] = "로봇 상태 복구 중..."
        robot_state["action"] = "recovery"

    proc = subprocess.Popen(
        ["bash", "-c", f"{ROS_SOURCE} && ros2 run cobot1 check_robot_state_recovery"],
        cwd=SCRIPT_DIR,
        env=env,
    )
    log.info("[RECOVERY] 프로세스 시작 PID=%d", proc.pid)
    pick_proc_manager.add(proc)
    threading.Thread(target=_monitor_recovery, args=(proc,), daemon=True).start()
