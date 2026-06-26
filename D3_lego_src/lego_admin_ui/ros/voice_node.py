import logging
import threading

from rclpy.node import Node
from std_msgs.msg import Bool

from ros.voice_state import voice_state, voice_lock
from process.process_manager import pick_proc_manager
from services.stop_service import do_stop_sync

log = logging.getLogger(__name__)


class VoiceNode(Node):
    """
    /pause_state 토픽을 구독해 음성 정지/재개 명령을 처리하는 ROS2 노드.

    처리 흐름:
        True 수신 (정지 키워드): voice_state 설정 → subprocess kill → 로봇 정지
        False 수신 (재개 키워드): voice_state 해제 → 자동화 루프 재개 허용
    """

    def __init__(self) -> None:
        super().__init__("voice_node")
        self.create_subscription(Bool, "/pause_state", self._callback, 10)
        log.info("[VOICE] VoiceNode 초기화 완료, /pause_state 구독 시작")

    def _callback(self, msg: Bool) -> None:
        if msg.data:
            with voice_lock:
                already = voice_state["paused"]
                voice_state["paused"] = True

            if not already:
                log.warning("[VOICE] 정지 키워드 감지 → 로봇 즉시 정지")
                pick_proc_manager.kill_all()
                threading.Thread(target=do_stop_sync, daemon=True).start()
        else:
            with voice_lock:
                voice_state["paused"] = False
            log.info("[VOICE] 재개 키워드 감지 → 자동화 재개 가능")
