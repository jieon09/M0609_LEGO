import logging
import subprocess
import threading

log = logging.getLogger(__name__)


class ProcessManager:
    """
    외부 subprocess 생명주기를 관리하는 레지스트리.

    스레드 안전한 목록으로 프로세스를 추적하고,
    일괄 종료(kill_all) 시 SIGTERM → 3초 대기 → SIGKILL 순서로 정리한다.
    """

    def __init__(self) -> None:
        self._procs: list[subprocess.Popen] = []
        self._lock: threading.Lock = threading.Lock()

    def add(self, proc: subprocess.Popen) -> None:
        """실행 중인 프로세스를 관리 목록에 등록한다."""
        with self._lock:
            self._procs.append(proc)

    def kill_all(self) -> None:
        """
        관리 중인 모든 프로세스를 종료하고 목록을 비운다.

        처리 흐름:
        1. SIGTERM 전송 (graceful shutdown 시도)
        2. 3초 대기 — ROS 노드 정리에 충분한 시간
        3. 3초 내 미종료 시 SIGKILL로 강제 제거

        Note:
            3초 timeout은 ROS 노드가 subscriber/publisher를 해제하는 데
            필요한 최소 시간을 기준으로 설정
        """
        with self._lock:
            for p in self._procs:
                if p.poll() is None:
                    log.debug("[PROC] PID=%d SIGTERM 전송", p.pid)
                    p.terminate()
                    try:
                        p.wait(timeout=3)
                        log.debug("[PROC] PID=%d 정상 종료", p.pid)
                    except subprocess.TimeoutExpired:
                        # 3초 내 종료 안 됨 → 강제 kill
                        log.warning("[PROC] PID=%d 3초 내 미종료 → SIGKILL", p.pid)
                        p.kill()
                else:
                    log.debug("[PROC] PID=%d 이미 종료됨 (returncode=%s)", p.pid, p.returncode)
            self._procs.clear()
            log.debug("[PROC] 프로세스 목록 초기화 완료")


# 모듈 수준 싱글톤 — pick 관련 서브프로세스를 전역에서 일괄 제어
pick_proc_manager = ProcessManager()
