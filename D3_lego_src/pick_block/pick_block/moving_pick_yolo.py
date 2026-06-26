#!/usr/bin/env python3

import os
import time
import sys
import threading
import numpy as np
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from ament_index_python.packages import get_package_share_directory
import DR_init

# ==========================================
# 패키지 및 전역 설정
# ==========================================
PACKAGE_NAME = "pick_block"
_here = os.path.dirname(os.path.abspath(__file__))

def _find_resource_dir() -> str:
    """
    T_gripper2camera.npy가 위치한 resource 디렉토리 경로를 반환

    처리 흐름:
    1. ament 패키지 share 디렉토리에서 resource 폴더 탐색
    2. 없으면 소스 기준 상대 경로로 폴백

    Returns:
        resource 디렉토리 절대 경로 (존재 여부 미보장)
    """
    try:
        candidate = os.path.join(get_package_share_directory(PACKAGE_NAME), "resource")
        if os.path.isdir(candidate):
            return candidate
    except Exception:
        pass
    for rel in ("resource", os.path.join("..", "resource")):
        candidate = os.path.normpath(os.path.join(_here, rel))
        if os.path.isdir(candidate):
            return candidate
    return os.path.join(_here, "resource")

GRIPPER2CAM_PATH = os.path.join(_find_resource_dir(), "T_gripper2camera.npy")

ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
VELOCITY, ACC = 100, 60  # 로봇 이동 속도/가속도 (%)
GRIPPER_NAME = "rg2"
TOOLCHANGER_IP = "192.168.1.1"   # OnRobot 툴체인저 IP
TOOLCHANGER_PORT = 502            # Modbus TCP 기본 포트

# 티치 펜던트에 설정된 Tool 및 TCP 이름
ROBOT_TOOL = "Tool Weight"
ROBOT_TCP = "GripperDA_v1"

# ==========================================
# 1. OnRobot 그리퍼 제어 클래스 (RG)
# ==========================================
from pymodbus.client.sync import ModbusTcpClient as ModbusClient

class RG:
    """OnRobot RG2/RG6 그리퍼 Modbus TCP 제어 (moving_pick_yolo 내부 전용)."""

    def __init__(self, gripper: str, ip: str, port: int) -> None:
        self.client = ModbusClient(ip, port=port, stopbits=1, bytesize=8, parity="E", baudrate=115200, timeout=1)
        if gripper not in ["rg2", "rg6"]:
            print("Please specify either rg2 or rg6.")
            return
        self.gripper = gripper
        if self.gripper == "rg2":
            self.max_width = 700  # RG2 최대 개방 너비 (0.1mm 단위)
            self.max_force = 400   # RG2 최대 파지력 (0.1N 단위)
        elif self.gripper == "rg6":
            self.max_width = 700  # RG6 최대 개방 너비 (0.1mm 단위)
            self.max_force = 400  # RG6 최대 파지력 (0.1N 단위)
        self.open_connection()

    def open_connection(self) -> None:
        """Modbus TCP 연결을 수립한다."""
        self.client.connect()

    def close_connection(self) -> None:
        """Modbus TCP 연결을 해제한다."""
        self.client.close()

    def close_gripper(self, force_val: int = 400) -> None:
        """
        그리퍼를 닫는다 (너비=0).

        Args:
            force_val: 파지력 (0.1N 단위, RG2 기준 최대 400)
        """
        params = [force_val, 0, 16]
        print("Start closing gripper.")
        self.client.write_registers(address=0, values=params, unit=65)

    def open_gripper(self, force_val: int = 400) -> None:
        """
        그리퍼를 최대 너비로 연다.

        Args:
            force_val: 파지력 (0.1N 단위, 열 때도 반력 기준으로 사용됨)
        """
        params = [force_val, self.max_width, 16]
        print("Start opening gripper.")
        self.client.write_registers(address=0, values=params, unit=65)

# ==========================================
# 2. 통합 로봇 제어 노드 (외부 데이터 수신)
# ==========================================
class ExternalPickAndPlaceNode(Node):
    """Flask 등 외부 프로세스로부터 픽셀/Depth 좌표를 받아 픽업 시퀀스를 실행하는 노드."""

    def __init__(self) -> None:
        super().__init__("external_pick_and_place")

        self.gripper = RG(GRIPPER_NAME, TOOLCHANGER_IP, TOOLCHANGER_PORT)
        self.init_robot()
        self.get_logger().info("External Pick & Place Node Initialized. Ready to receive commands.")

    def get_robot_pose_matrix(
        self,
        x: float, y: float, z: float,   # robot base 좌표계, mm 단위
        rx: float, ry: float, rz: float, # ZYZ 오일러 각도, deg 단위
    ) -> np.ndarray:
        """
        로봇 TCP 포즈(posx)를 4x4 동차 변환 행렬로 변환

        Args:
            x, y, z: 로봇 베이스 좌표계 위치 (mm)
            rx, ry, rz: ZYZ 오일러 각도 (deg)

        Returns:
            (4, 4) 동차 변환 행렬 [R|t; 0 1]
        """
        R = Rotation.from_euler("ZYZ", [rx, ry, rz], degrees=True).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        return T

    def transform_pose_to_base(
        self,
        camera_coords: list[float],  # 카메라 좌표계 [x, y, z], mm 단위
        angle_deg: float,            # 블럭 검출 각도, deg 단위
        robot_pos: list[float],      # 현재 TCP posx [x,y,z,rx,ry,rz]
    ) -> tuple[np.ndarray, list[float]]:
        """
        카메라 좌표계 3D 점을 로봇 베이스 좌표계로 변환하고 그리퍼 방향을 보정

        처리 흐름:
        1. 파일에서 gripper→camera 변환 행렬(T_gripper2camera) 로드
        2. 현재 TCP posx로 base→gripper 변환 행렬 구성
        3. base→cam = base→gripper @ gripper→cam 으로 3D 점 변환
        4. 블럭 검출 각도(angle_deg)를 카메라 Z축 회전으로 합성해 그리퍼 자세 보정

        Args:
            camera_coords: 카메라 좌표계 3D 좌표 [x, y, z] (mm)
            angle_deg: 블럭의 카메라 기준 회전각 (deg)
            robot_pos: 현재 TCP posx 6-DOF [x,y,z,rx,ry,rz]

        Returns:
            (base_xyz, [rx, ry, rz]): 베이스 좌표계 위치(mm)와 ZYZ 오일러 각도(deg)
        """
        gripper2cam = np.load(GRIPPER2CAM_PATH)
        coord = np.append(np.array(camera_coords), 1)

        x, y, z, rx, ry, rz = robot_pos
        base2gripper = self.get_robot_pose_matrix(x, y, z, rx, ry, rz)

        base2cam = base2gripper @ gripper2cam
        td_coord = np.dot(base2cam, coord)[:3]

        R_base2gripper = base2gripper[:3, :3]
        R_gripper2cam = gripper2cam[:3, :3]

        # 블럭 각도를 카메라 Z축 회전으로 변환하여 그리퍼 목표 자세에 합성
        R_cam_rot = Rotation.from_euler('z', angle_deg, degrees=True).as_matrix()
        R_base2gripper_new = R_base2gripper @ R_gripper2cam @ R_cam_rot @ R_gripper2cam.T

        new_rx, new_ry, new_rz = Rotation.from_matrix(R_base2gripper_new).as_euler('ZYZ', degrees=True)

        return td_coord, [new_rx, new_ry, new_rz]

    def init_robot(self) -> None:
        """
        Tool/TCP 설정 후 로봇을 초기 조인트 자세(JReady)로 이동

        Note:
            set_tool/set_tcp는 AUTONOMOUS 모드에서 블로킹되므로
            MANUAL 모드로 전환 후 설정하고 AUTONOMOUS로 복귀한다.
            JReady는 펌웨어 33버전 기준 실측값 (단위: deg).
        """
        try:
            from DSR_ROBOT2 import ROBOT_MODE_MANUAL, ROBOT_MODE_AUTONOMOUS, set_robot_mode
            set_robot_mode(ROBOT_MODE_MANUAL)
            time.sleep(0.5)
            set_tool(ROBOT_TOOL)
            set_tcp(ROBOT_TCP)
            set_robot_mode(ROBOT_MODE_AUTONOMOUS)
            time.sleep(2.0)
            self.get_logger().info(f"Tool & TCP 설정 완료. Tool: {ROBOT_TOOL}, TCP: {ROBOT_TCP}")
        except Exception as e:
            self.get_logger().warn(f"Tool/TCP 설정 실패: {e}")

        JReady = [11.07, 1.16, 88.84, 0, 90.01, 11.07]  # 초기 안전 자세 (deg)
        movej(JReady, vel=VELOCITY, acc=ACC)
        mwait()

    # =========================================================
    # 외부 입력값을 사용한 핵심 이동 및 픽업 시퀀스
    # =========================================================
    def execute_pick_sequence_from_external(
        self,
        cx: float,        # 블럭 중심 픽셀 좌표 X (pixel)
        cy: float,        # 블럭 중심 픽셀 좌표 Y (pixel)
        cz: float,        # 블럭까지의 depth (mm)
        angle_deg: float, # 블럭 검출 각도 (deg)
        fx: float,        # 카메라 초점거리 X (pixel)
        fy: float,        # 카메라 초점거리 Y (pixel)
        ppx: float,       # 카메라 주점 X (pixel)
        ppy: float,       # 카메라 주점 Y (pixel)
    ) -> bool:
        """
        외부(Flask 등)에서 전달된 픽셀/Depth 좌표로 픽업 시퀀스 실행

        처리 흐름:
        1. 핀홀 모델로 픽셀+Depth → 카메라 좌표계 3D 점 변환
        2. 현재 TCP posx 획득 후 베이스 좌표계로 변환
        3. 그리퍼 열기 → XY 정렬 이동
        4. 순응 제어 On → 177mm 수직 하강 → 그리퍼 닫기
        5. 순응 제어 Off → 20mm 상승 → 하중(Fz) 측정으로 파지 검증
        6. 파지 성공 시 안전 높이로 복귀, 실패 시 그리퍼 열고 복귀

        Args:
            cx, cy: 이미지 내 블럭 중심 픽셀 좌표
            cz: depth 값 (mm)
            angle_deg: 블럭 방향 각도 (deg)
            fx, fy: 카메라 초점거리 (pixel 단위)
            ppx, ppy: 카메라 주점 좌표 (pixel 단위)

        Returns:
            True: 파지 성공, False: 파지 실패
        """
        Y_OFFSET = 0  # 필요 시 Y축 오프셋 보정치 입력 (mm, ex: -18)

        self.get_logger().info(f"외부 명령 수신 -> 픽셀:({cx}, {cy}), Depth:{cz}mm, Angle:{angle_deg}°")

        # 핀홀 모델로 픽셀+Depth를 카메라 좌표계 3D 점으로 변환 (단위: mm)
        # Flask 단에서 이미 3D 좌표를 계산해 전달하는 경우 이 변환은 생략 가능
        cam_x_c = (cx - ppx) * cz / fx
        cam_y_c = (cy - ppy) * cz / fy
        c3d_coords = [cam_x_c, cam_y_c, cz]

        # 현재 TCP 위치를 기준으로 카메라 좌표 → 로봇 베이스 좌표 변환
        robot_pos_start = get_current_posx()[0]
        base_target, target_ori = self.transform_pose_to_base(c3d_coords, angle_deg, robot_pos_start)

        final_x = base_target[0]          # robot base 좌표계, mm
        final_y = base_target[1] + Y_OFFSET  # robot base 좌표계, mm
        curr_z = robot_pos_start[2]       # 현재 로봇 Z 높이 유지 (depth 값 미사용), mm

        self.gripper.open_gripper()

        # 목표 XY 좌표 상공으로 이동 (Z 높이 유지)
        align_pos = [final_x, final_y, curr_z] + list(target_ori)
        self.get_logger().info("Step 1: 목표 X, Y 좌표 상공으로 정렬 이동")
        movel(align_pos, vel=VELOCITY, acc=ACC)
        mwait()

        # 순응 제어 On 후 177mm 수직 하강
        # stx Z축 강성 800: 접촉 시 과부하를 방지하기 위해 XY보다 낮게 설정
        target_z = curr_z - 177.0  # 실측 기준 블럭 파지 높이까지의 하강 거리 (mm)
        final_pos_lower = [final_x, final_y, target_z] + list(target_ori)

        self.get_logger().info("Step 2: 수직 하강 시작 (순응 제어 On)")
        task_compliance_ctrl(stx=[3000, 3000, 800, 200, 200, 200])
        time.sleep(0.2)
        movel(final_pos_lower, vel=VELOCITY, acc=ACC)
        mwait()

        self.get_logger().info("Step 3: 그리퍼 close")
        self.gripper.close_gripper(force_val=400)
        time.sleep(1.0)  # 그리퍼 동작 완료 대기 (모터 응답 지연 고려)

        # 순응 제어 해제 후 미세 상승: 하중 측정은 강체 상태에서 정확하므로 해제 필요
        self.get_logger().info("Step 4: 순응 제어 해제 및 하중 검증을 위한 미세 상승")
        release_compliance_ctrl()
        time.sleep(0.1)  # 순응 제어 해제 반영 대기

        measure_pos = [final_x, final_y, target_z + 20.0] + list(target_ori)
        movel(measure_pos, vel=VELOCITY, acc=ACC)
        mwait()
        time.sleep(0.5)  # 진동 수렴 대기 후 하중 측정

        # Fz 하중으로 파지 여부 검증 (단위: N)
        current_force = get_tool_force(0)
        fz_weight = abs(current_force[2])
        self.get_logger().info(f"측정된 툴단 하중(Fz): {fz_weight:.2f} N")

        WEIGHT_THRESHOLD = 0.1  # 파지 판정 임계값 (N); 블럭 무게 기반 실측값
        if fz_weight >= WEIGHT_THRESHOLD:
            self.get_logger().info("파지 성공 확인.")
        else:
            self.get_logger().warn("파지 실패 감지. 그리퍼를 열고 시퀀스를 복귀합니다.")
            self.gripper.open_gripper()
            movel(align_pos, vel=VELOCITY, acc=ACC)
            mwait()
            return False

        self.get_logger().info("Step 5: 원래 안전 높이(Z축)로 복귀")
        movel(align_pos, vel=VELOCITY, acc=ACC)
        mwait()

        self.get_logger().info("픽업 시퀀스 최종 완료.")
        return True

# ==========================================
# 3. 메인 실행 블록
# ==========================================
def main(args: list | None = None) -> None:
    """
    커맨드라인 인자로 픽셀/Depth 좌표를 받아 픽업 시퀀스를 실행

    처리 흐름:
    1. argv에서 cx cy cz angle_deg fx fy ppx ppy 파싱
    2. ROS2 초기화 및 DSR 노드 생성
    3. DSR_ROBOT2 API import
    4. ExternalPickAndPlaceNode 생성 후 픽업 시퀀스 실행
    5. 성공 여부에 따라 exit code 반환

    Note:
        Flask 서버 등 외부 프로세스가 subprocess로 호출하는 진입점이다.
        blocking 실행이므로 타임아웃 관리는 호출 측에서 담당해야 한다.
    """
    if len(sys.argv) < 9:
        print(f"Usage: {sys.argv[0]} cx cy cz angle_deg fx fy ppx ppy")
        sys.exit(1)

    cx, cy, cz, angle_deg, fx, fy, ppx, ppy = map(float, sys.argv[1:9])

    rclpy.init(args=args)

    # DR_init에 노드를 연결해야 DSR_ROBOT2 내부 ROS 통신이 정상 동작한다
    DR_init.__dsr__id = ROBOT_ID
    DR_init.__dsr__model = ROBOT_MODEL
    dsr_node = rclpy.create_node("pick_block", namespace=ROBOT_ID)
    DR_init.__dsr__node = dsr_node

    # DSR_ROBOT2 서비스/액션 콜백 처리를 위한 spin 스레드 (없으면 mwait 등이 블로킹됨)
    # spin_once 루프 방식: _spin_stop 이벤트로 안전하게 종료 가능 (SIGABRT 방지)
    _executor = SingleThreadedExecutor()
    _executor.add_node(dsr_node)
    _spin_stop = threading.Event()

    def _spin_loop():
        while not _spin_stop.is_set():
            _executor.spin_once(timeout_sec=0.05)

    _spin_thread = threading.Thread(target=_spin_loop, daemon=True)
    _spin_thread.start()

    try:
        global movej, movel, get_current_posx, mwait, trans, task_compliance_ctrl, release_compliance_ctrl, get_tool_force, set_tool, set_tcp
        from DSR_ROBOT2 import movej, movel, get_current_posx, mwait, trans, task_compliance_ctrl, release_compliance_ctrl, get_tool_force, set_tool, set_tcp
    except ImportError as e:
        print(f"Error importing DSR_ROBOT2: {e}")
        sys.exit(1)

    node = ExternalPickAndPlaceNode()
    exit_code = 0
    try:
        success = node.execute_pick_sequence_from_external(
            cx=cx, cy=cy, cz=cz, angle_deg=angle_deg,
            fx=fx, fy=fy, ppx=ppx, ppy=ppy,
        )
        if not success:
            node.get_logger().warn("픽업 시퀀스 실패.")
            exit_code = 1
    except KeyboardInterrupt:
        node.get_logger().info("프로그램을 종료합니다.")
        exit_code = 1
    except Exception as e:
        node.get_logger().error(f"픽업 시퀀스 예외 발생: {e}")
        exit_code = 1

    # 그리퍼 연결만 닫고 즉시 종료 — destroy_node/rclpy.shutdown 호출 시
    # spin 스레드와의 경쟁으로 C++ SIGABRT(-6)가 발생해 exit_code가 무시됨
    try:
        node.gripper.close_connection()
    except Exception:
        pass
    os._exit(exit_code)

if __name__ == "__main__":
    main()