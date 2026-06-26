"""
pause_control/robot_node.py
────────────────────────────────────────────────────────────────────────────
ROS 2 노드: /pause_state 구독 → threading.Event 로 로봇 작업 제어

구독 토픽: /pause_state  (std_msgs/Bool)
    True  → pause_event.set()   (동작 재개)
    False → pause_event.clear() (일시 정지)

실행:
    ros2 run pause_control robot_node
"""

import sys
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
import DR_init
from DSR_ROBOT2 import movej, mwait

# ─── 로봇 설정 ───────────────────────────────────────────────────────────────
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
VELOCITY = 30
ACC = 30

# ─── 동작 자세 정의 ──────────────────────────────────────────────────────────
HOME = [0, 0, 90, 0, 90, 0]
POS_B = [30, 0, 90, 0, 90, 0]
POS_C = [-30, 0, 90, 0, 90, 0]

# ─── 공유 이벤트 ─────────────────────────────────────────────────────────────
# set   → 정상 동작  (pause_event.wait() 즉시 통과)
# clear → 일시 정지  (pause_event.wait() 에서 블록)
pause_event = threading.Event()
pause_event.set()


class RobotNode(Node):

    def __init__(self):
        super().__init__("robot_node")

        DR_init.__dsr__id = ROBOT_ID
        DR_init.__dsr__model = ROBOT_MODEL
        DR_init.__dsr__node = self

        self.create_subscription(Bool, "/pause_state", self._pause_callback, 10)
        self.get_logger().info("RobotNode 시작 | /pause_state 구독 중")

    def _pause_callback(self, msg: Bool):
        if msg.data:
            pause_event.set()
            self.get_logger().info("▶  재개 수신")
        else:
            pause_event.clear()
            self.get_logger().info("⏸  정지 수신")


# ─── 로봇 작업 함수 ───────────────────────────────────────────────────────────
def go_home():
    print("[로봇] 홈 복귀 중...")
    movej(HOME, vel=VELOCITY, acc=ACC)
    mwait()
    print("[로봇] 홈 복귀 완료 - 재개 명령 대기 중...")


def do_move(label, pos):
    print(f"[로봇] {label} 이동 중...")
    movej(pos, vel=VELOCITY, acc=ACC)
    mwait()
    print(f"[로봇] {label} 완료")

    # ── 체크포인트: 정지 명령이 왔으면 홈 복귀 후 대기 ──────────────────
    if not pause_event.is_set():
        go_home()
        pause_event.wait()
        print("[로봇] 재개 - 다음 동작 계속합니다")


# ─── 메인 ────────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = RobotNode()

    # Subscriber spin 은 별도 스레드, 메인 스레드에서 작업 루프 실행
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    print("[로봇] 준비 완료!")
    print("  '잠깐 멈춰줘' → 현재 동작 마치고 홈 복귀 후 정지")
    print("  '다시 실행해' → 재개\n")

    movej(HOME, vel=VELOCITY, acc=ACC)
    mwait()

    try:
        cycle = 1
        while True:
            print(f"\n── 사이클 {cycle} ──────────────────")
            do_move("POS_B (오른쪽)", POS_B)
            do_move("POS_C (왼쪽)", POS_C)
            do_move("HOME  (홈)", HOME)
            cycle += 1
    except KeyboardInterrupt:
        print("\n[로봇] 종료")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
