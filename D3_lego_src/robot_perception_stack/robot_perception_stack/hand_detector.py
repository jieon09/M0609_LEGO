import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image          # 카메라 이미지 메시지 타입
from std_msgs.msg import Bool              # 손 감지 결과 (True/False)
from cv_bridge import CvBridge             # ROS Image ↔ OpenCV 변환

import cv2
import mediapipe as mp

class HandDetector(Node):
    def __init__(self):
        # ROS2 노드 초기화 (노드 이름: hand_detector)
        super().__init__('hand_detector')

        # ─────────────────────────────────────────────
        # CvBridge: ROS Image → OpenCV 이미지 변환용
        # frame: 최신 카메라 프레임 저장 변수
        # ─────────────────────────────────────────────
        self.bridge = CvBridge()
        self.frame = None

        # ─────────────────────────────────────────────
        # 손 감지 결과 publish
        # topic: /hand_detected
        # type: Bool (True = 손 있음, False = 없음)
        # ─────────────────────────────────────────────
        self.pub = self.create_publisher(Bool, '/hand_detected', 10)

        # ─────────────────────────────────────────────
        # 카메라 이미지 subscribe
        # topic: /camera/camera/color/image_raw
        # ─────────────────────────────────────────────
        self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10
        )

        # ─────────────────────────────────────────────
        # MediaPipe Hands 초기화
        # - static_image_mode=False → 영상 스트림 모드
        # - max_num_hands=2 → 최대 2개 손 인식
        # - confidence → 검출/추적 정확도 설정
        # ─────────────────────────────────────────────
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # ─────────────────────────────────────────────
        # 30Hz 주기로 손 감지 수행
        # ─────────────────────────────────────────────
        self.create_timer(1.0 / 30.0, self.detect_loop)

        self.get_logger().info("Hand Detector Node Started")

    # ─────────────────────────────────────────────
    # 카메라 콜백 함수
    # - ROS 이미지 메시지를 OpenCV 이미지로 변환해서 저장
    # ─────────────────────────────────────────────
    def image_callback(self, msg: Image):
        try:
            self.frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")

    # ─────────────────────────────────────────────
    # 메인 감지 루프 (30Hz)
    # - 최신 프레임을 가져와서 MediaPipe로 손 검출
    # - 결과를 Bool 토픽으로 publish
    # ─────────────────────────────────────────────
    def detect_loop(self):
        if self.frame is None:
            return  # 아직 카메라 프레임 없음

        # 현재 프레임 복사 (thread safety)
        frame = self.frame.copy()

        # OpenCV BGR → RGB 변환 (MediaPipe 입력 형식)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # MediaPipe 손 검출 수행
        result = self.hands.process(rgb)

        # 손이 하나라도 감지되면 True
        hand_detected = result.multi_hand_landmarks is not None

        # ─────────────────────────────────────────
        # ROS publish (핵심 output)
        # ─────────────────────────────────────────
        msg = Bool()
        msg.data = hand_detected
        self.pub.publish(msg)

        # ─────────────────────────────────────────
        # 로그 출력 (디버깅용)
        # ─────────────────────────────────────────
        if hand_detected:
            self.get_logger().warn("✋ HAND DETECTED")
        else:
            self.get_logger().info("SAFE")


# ─────────────────────────────────────────────
# ROS2 실행 entry point
# ─────────────────────────────────────────────
def main():
    rclpy.init()
    node = HandDetector()

    try:
        rclpy.spin(node)  # 콜백 계속 실행
    except KeyboardInterrupt:
        pass
    finally:
        # MediaPipe 리소스 정리
        node.hands.close()

        # ROS 노드 정리
        node.destroy_node()
        rclpy.shutdown()


# ─────────────────────────────────────────────
# 직접 실행 시 main() 실행
# ─────────────────────────────────────────────
if __name__ == '__main__':
    main()