import threading
import math
from typing import Any

from config import ROBOT_ID
# 로봇이 뭘 하고 있는지 UI에 알려주기 위한 상태 메시지 업데이트 log
# 실제 로봇 동작과는 무관하고 순수하게 표시용입니다
# 로봇 상태 공유 딕셔너리 — 여러 스레드에서 접근하므로 반드시 state_lock으로 보호
robot_state: dict[str, Any] = {
    # 관절각: robot base 좌표계, deg 단위
    "joints": {"j1": 0.0, "j2": 0.0, "j3": 0.0, "j4": 0.0, "j5": 0.0, "j6": 0.0},
    # TCP 위치: robot base 좌표계, mm(x,y,z) / deg(rx,ry,rz) 단위
    "tcp":    {"x": 0.0, "y": 0.0, "z": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
    "status": "초기화 중",
    "action": None,               # 현재 실행 중인 동작 이름 (pick/place/None)
    "ros_connected": False,
    "safety_mode": False,
    "automation_running": False,  # 자동화 루프가 Task를 처리 중인지 여부
    "automation_cancel":  False,  # True 설정 시 진행 중인 Task를 중단 요청
    "robot_hw_state": -1,         # 실제 하드웨어 상태 코드 (GetRobotState 서비스 응답값)
                                  # -1: 미수신, 1: STANDBY, 3: SAFE_OFF, 5: SAFE_STOP 등
}

# robot_state 접근 시 반드시 사용해야 하는 전역 락
state_lock = threading.Lock()


def _ros_spin_thread() -> None:
    """
    ROS2 JointState 구독 및 TCP 포즈 폴링을 담당하는 백그라운드 스레드 함수.

    처리 흐름:
    1. rclpy / DSR 메시지 import (ROS 미설치 환경 대비)
    2. RobotMonitorNode 생성 — JointState 구독 + 0.1초 TCP 폴링 타이머
    3. rclpy.spin() 진입 (블로킹)
    4. 오류 시 ros_connected=False로 갱신 후 종료

    Note:
        이 함수는 daemon 스레드에서 실행되므로
        메인 프로세스 종료 시 join 없이 자동 회수된다.
    """
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
        from dsr_msgs2.srv import GetCurrentPose, GetRobotState
    except Exception as e:
        with state_lock:
            robot_state["status"] = f"ROS 임포트 실패: {e}"
        return

    try:
        if not rclpy.ok():
            rclpy.init()
    except Exception as e:
        with state_lock:
            robot_state["status"] = f"ROS 초기화 실패: {e}"
        return

    class RobotMonitorNode(Node):
        """
        로봇 관절각·TCP 포즈를 수신해 robot_state를 갱신하는 내부 노드.

        Note:
            _on_joint_states: ROS 콜백 — 무거운 연산 금지 (콜백 점유 시 다음 메시지 지연)
            _request_tcp_pose: 0.1초 타이머 — 서비스 미준비/pending 상태 보호
        """

        def __init__(self) -> None:
            super().__init__("lego_admin_ui_node")

            self.create_subscription(
                JointState,
                f"/{ROBOT_ID}/joint_states",
                self._on_joint_states,
                10,
            )

            self._pose_cli = self.create_client(
                GetCurrentPose,
                f"/{ROBOT_ID}/system/get_current_pose",
            )

            # 0.1초(10 Hz) 주기로 TCP 포즈 서비스 호출 — UI 갱신에 충분한 빈도
            self.create_timer(0.1, self._request_tcp_pose)
            self._pending_future = None

            # 1초 주기로 하드웨어 상태 코드 폴링
            self._state_cli = self.create_client(
                GetRobotState,
                f"/{ROBOT_ID}/system/get_robot_state",
            )
            self.create_timer(1.0, self._request_robot_state)
            self._pending_state_future = None

            with state_lock:
                robot_state["ros_connected"] = True
                robot_state["status"] = "대기 중"

        def _on_joint_states(self, msg: Any) -> None:
            """
            JointState 메시지를 수신해 관절각을 deg로 변환 후 robot_state 갱신.

            Note:
                ROS 콜백이므로 처리가 빠르게 끝나야 함.
                rad → deg 변환 외 무거운 연산 금지.
            """
            name_to_pos: dict[str, float] = dict(zip(msg.name, msg.position))
            with state_lock:
                js = robot_state["joints"]
                for i, key in enumerate(["j1", "j2", "j3", "j4", "j5", "j6"], 1):
                    rad: float = name_to_pos.get(f"joint_{i}", 0.0)
                    js[key] = round(math.degrees(rad), 2)  # rad → deg

        def _request_tcp_pose(self) -> None:
            """
            TCP 포즈 서비스를 비동기 호출한다.

            Note:
                이전 요청이 아직 처리 중이면 중복 호출을 건너뜀.
                space_type=1: task 공간(TCP) 좌표 요청.
            """
            if not self._pose_cli.service_is_ready():
                return
            # 이전 요청 미완료 시 중복 호출 방지
            if self._pending_future and not self._pending_future.done():
                return
            req = GetCurrentPose.Request()
            req.space_type = 1  # 1: task 공간 (TCP) 좌표
            self._pending_future = self._pose_cli.call_async(req)
            self._pending_future.add_done_callback(self._on_tcp_pose)

        def _on_tcp_pose(self, future: Any) -> None:
            """
            TCP 포즈 서비스 응답을 받아 robot_state["tcp"]를 갱신한다.

            Note:
                pos[0~2]: x,y,z (mm, robot base 좌표계)
                pos[3~5]: rx,ry,rz (deg, robot base 좌표계)
            """
            try:
                res = future.result()
                if res.success and len(res.pos) >= 6:
                    with state_lock:
                        tcp = robot_state["tcp"]
                        tcp["x"]  = round(res.pos[0], 2)
                        tcp["y"]  = round(res.pos[1], 2)
                        tcp["z"]  = round(res.pos[2], 2)
                        tcp["rx"] = round(res.pos[3], 2)
                        tcp["ry"] = round(res.pos[4], 2)
                        tcp["rz"] = round(res.pos[5], 2)
            except Exception:
                pass

        def _request_robot_state(self) -> None:
            """1초 주기로 하드웨어 상태 코드를 폴링한다."""
            if not self._state_cli.service_is_ready():
                return
            if self._pending_state_future and not self._pending_state_future.done():
                return
            self._pending_state_future = self._state_cli.call_async(
                GetRobotState.Request()
            )
            self._pending_state_future.add_done_callback(self._on_robot_state)

        def _on_robot_state(self, future: Any) -> None:
            """GetRobotState 응답을 받아 robot_hw_state를 갱신한다."""
            try:
                res = future.result()
                with state_lock:
                    robot_state["robot_hw_state"] = int(res.robot_state)
            except Exception:
                pass

    try:
        node = RobotMonitorNode()
        rclpy.spin(node)
        node.destroy_node()
    except Exception as e:
        with state_lock:
            robot_state["ros_connected"] = False
            robot_state["status"] = f"ROS 연결 오류: {e}"
    finally:
        if rclpy.ok():
            rclpy.shutdown()


# 모듈 import 시 즉시 기동 — daemon=True로 메인 종료 시 자동 회수
ros_thread = threading.Thread(target=_ros_spin_thread, daemon=True)
ros_thread.start()
