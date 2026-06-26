import time
import rclpy
from rclpy.node import Node

import DR_init

ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

ROBOT_TOOL = "Tool Weight"
ROBOT_TCP = "GripperDA_v1"

# 홈 관절 각도: deg 단위, robot base 좌표계
HOME_POSJ = [0.0, 0.0, 90.0, 0.0, 90.0, 0.0]

class GoHomeNode(Node):
    def __init__(self):
        # namespace=ROBOT_ID 로 만들어 DSR 서비스(/dsr01/...) 와 매칭시킨다.
        super().__init__("go_home_node", namespace=ROBOT_ID)

        # DSR_ROBOT2 import 시 g_node = DR_init.__dsr__node 를 캡처하므로,
        # 반드시 import 전에 노드를 할당해야 한다.
        DR_init.__dsr__node = self

        self.get_logger().info("go_home")
        # 초기화하기
        self._init_robot()

    def _init_robot(self) -> None:
        """
        TCP / Tool 설정 후 HOME 위치로 이동한다.

        처리 흐름:
        1. MANUAL 모드 전환
        2. TCP / Tool 설정
        3. AUTONOMOUS 모드 복귀
        4. HOME_POSJ 위치로 movej 이동
        """

        self.get_logger().info("Initializing robot...")

        from DSR_ROBOT2 import (
            set_tool,
            set_tcp,
            movej,
            mwait,
            ROBOT_MODE_MANUAL,
            ROBOT_MODE_AUTONOMOUS,
            set_robot_mode,
        )

        # 1. MANUAL 전환
        set_robot_mode(ROBOT_MODE_MANUAL)
        time.sleep(0.5)

        # 2. TCP / TOOL 설정
        set_tool(ROBOT_TOOL)
        set_tcp(ROBOT_TCP)

        # 3. AUTONOMOUS 복귀
        set_robot_mode(ROBOT_MODE_AUTONOMOUS)
        time.sleep(1.0)

        # 4. HOME 이동
        self.get_logger().info(f"Moving HOME: {HOME_POSJ}")
        movej(HOME_POSJ, vel=60, acc=50)
        mwait()

        self.get_logger().info("Robot Init Complete")


def main(args=None):
    rclpy.init(args=args)

    node = GoHomeNode()

    try:
        rclpy.spin_once(node, timeout_sec=0.5)  # 1회만 실행 느낌
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()