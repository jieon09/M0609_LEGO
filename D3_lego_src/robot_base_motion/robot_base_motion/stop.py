import rclpy
from rclpy.node import Node
from dsr_msgs2.srv import MoveStop

class StopNode(Node):
    def __init__(self):
        super().__init__('stop_node')

        self.client = self.create_client(
            MoveStop,
            '/dsr01/motion/move_stop'
        )

        self.get_logger().info("Waiting for stop service...")

        if not self.client.wait_for_service(timeout_sec=3.0):
            self.get_logger().error("Stop service not available")
            return

        self.get_logger().info("Stop service ready")

    def request_stop(self, mode: int = 1) -> None:
        """
        로봇 정지 서비스를 비동기 요청한다.

        Args:
            mode: 정지 방식 (1 = 일반 stop)
        """
        req = MoveStop.Request()
        req.stop_mode = mode

        future = self.client.call_async(req)
        future.add_done_callback(self._on_stop_response)

    def _on_stop_response(self, future) -> None:
        try:
            result = future.result()
            self.get_logger().info(f"Stop success: {result}")
        except Exception as e:
            self.get_logger().error(f"Stop failed: {e}")


def main(args=None):
    rclpy.init(args=args)

    node = StopNode()

    node.request_stop(mode=1)

    rclpy.spin_once(node, timeout_sec=1.0)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()