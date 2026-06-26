import logging
import threading

import rclpy
from rclpy.executors import MultiThreadedExecutor

from ros.camera_node import CameraNode
from ros.hand_node import HandNode
from ros.voice_node import VoiceNode

log = logging.getLogger(__name__)


def _run() -> None:
    """
    ROS2 카메라·손감지 노드를 초기화하고 spin하는 내부 함수.

    처리 흐름:
    1. rclpy.init() — 이미 초기화된 경우 재시도 건너뜀
    2. CameraNode·HandNode 생성 및 MultiThreadedExecutor에 등록
    3. spin() — 토픽 수신 루프 진입 (블로킹)
    4. 종료 시 노드 정리 및 rclpy.shutdown()

    Note:
        MultiThreadedExecutor를 사용하는 이유: 컬러·깊이·카메라정보 세 토픽이
        동시에 들어올 때 순차 처리로 인한 지연을 줄이기 위함.
        HandNode를 같은 executor에 등록해 rclpy.init() 중복 호출을 피한다.
    """
    try:
        if not rclpy.ok():
            rclpy.init()
        log.info("[CAM] rclpy.init() 완료")
    except Exception as e:
        log.error("[CAM] rclpy.init() 실패: %s", e)
        return

    try:
        camera_node = CameraNode()
        hand_node   = HandNode()
        voice_node  = VoiceNode()
        executor    = MultiThreadedExecutor()
        executor.add_node(camera_node)
        executor.add_node(hand_node)
        executor.add_node(voice_node)
        log.info("[CAM] 노드 생성 완료 (CameraNode + HandNode + VoiceNode), spin 시작")
        executor.spin()
        camera_node.destroy_node()
        hand_node.destroy_node()
        voice_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        log.info("[CAM] spin 종료, 노드 정리 완료")
    except Exception as e:
        log.error("[CAM] 노드 오류: %s", e, exc_info=True)


def start_camera_thread() -> None:
    """
    카메라 노드를 daemon 스레드로 기동한다.

    Note:
        daemon=True로 설정해 Flask 메인 프로세스 종료 시 스레드가 자동 회수됨.
        별도 join 없이 앱 종료가 가능한 이유.
    """
    threading.Thread(target=_run, daemon=True).start()
