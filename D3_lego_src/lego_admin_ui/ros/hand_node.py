import logging
import threading

from rclpy.node import Node
from std_msgs.msg import Bool

from ros.hand_state import hand_state, hand_lock
from process.process_manager import pick_proc_manager
from services.stop_service import do_stop_sync

log = logging.getLogger(__name__)


class HandNode(Node):
    """
    /hand_detected 토픽을 구독해 손 감지 시 로봇을 즉시 정지시키는 ROS2 노드.

    처리 흐름:
        True 수신: hand_state 설정 → 실행 중 subprocess kill → 로봇 정지 서비스 호출
        False 수신: hand_state 해제 → 자동화 루프 재개 허용

    Note:
        do_stop_sync()는 블로킹이므로 콜백 스레드를 점유하지 않도록 daemon 스레드로 분리.
        kill_all()은 pick/place/move subprocess를 즉시 종료해 모션을 중단시킨다.
    """

    def __init__(self) -> None:
        super().__init__("hand_node")
        self.create_subscription(Bool, "/hand_detected", self._callback, 10)
        log.info("[HAND] HandNode 초기화 완료, /hand_detected 구독 시작")

    def _callback(self, msg: Bool) -> None:
        """
        /hand_detected 메시지 수신 시 호출되는 콜백.

        Args:
            msg: True = 손 감지, False = 손 없음
        """
        if msg.data:
            with hand_lock:
                already = hand_state["detected"]
                hand_state["detected"] = True

            if not already:
                log.warning("[HAND] 손 감지 → 로봇 즉시 정지")
                pick_proc_manager.kill_all()
                # do_stop_sync는 blocking이므로 콜백 스레드 블록 방지를 위해 분리
                threading.Thread(target=do_stop_sync, daemon=True).start()
        else:
            with hand_lock:
                hand_state["detected"] = False            
            #log.info("[HAND] 손 감지 해제 → 자동화 재개 가능")
