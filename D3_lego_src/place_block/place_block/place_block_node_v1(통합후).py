#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
흐름:
1) config/blocks.json 또는 파라미터 읽기
2) config/board_calibration.yaml 읽기
3) JSON의 row/col/width/height를 24x24 실제 board 좌표로 변환
4) 변환된 target posx를 기준으로 접근 -> 하강 -> 그리퍼 보정 시퀀스 -> 상승
5) dry_run=True이면 실제 로봇은 움직이지 않고 계산 좌표만 출력

좌표로 이동하는 시퀀스 (v15_3 및 node16 반영, 자동 무인 실행 버전):
1) 중심 좌표로 이동 (cell_center_approach)
2) 대각선 이동 (diagonal_offset)
3) 각도 보정 (rotate_before_insert)
4) z축 내려가기 (target_down)
5) 그리퍼 열기(50mm) -> 상승 -> 25mm로 변경 -> 하강 -> 상승 복귀 (release_sequence)
"""

import os
import sys
import json
import time
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import yaml
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
import DR_init  # type: ignore

from ament_index_python.packages import get_package_share_directory
from place_block.onrobot import RG  # type: ignore


# ============================================================
# Robot / Package 설정
# ============================================================
PACKAGE_NAME = "place_block"
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

ROBOT_TOOL = "Tool Weight"
ROBOT_TCP = "GripperDA_v1"

# DSR_ROBOT2 전역 매핑
movej = None
movel = None
mwait = None
ikin = None
task_compliance_ctrl = None
release_compliance_ctrl = None
set_stiffnessx = None

# ============================================================
# 상숫값 및 하드코딩 백업 설정
# ============================================================
LAYER_INSERT_Z = {
    1: 35.00,
    2: 49.299,
    3: 68.4,
}

HOME_POSX = [367.445, 6.569, 193.709, 172.213, 179.892, 172.226]
HOME_POSJ = [-0.003, 0.014, 90.097, 0.004, 89.997, 0.07]

DEFAULT_INSERT_OFFSET_X_MM = 10.5
DEFAULT_INSERT_OFFSET_Y_MM = -7.5

INSERT_2X3_OFFSET_X_MM = 10.5
INSERT_2X3_OFFSET_Y_MM = -19.5

COLUMN_CORRECTION_ENABLED = True
COLUMN_CORRECTION_REFERENCE_COL = 14
COLUMN_CORRECTION_X_MM = 2.0
COLUMN_CORRECTION_Y_MM = 0.0
COLUMN_CORRECTION_MODE = "greater_equal"

DEFAULT_GRIPPER_C_OFFSET_DEG = +6.5
INSERT_VERTICAL_A_DEG = 80.563
INSERT_VERTICAL_B_DEG = -178.883

CONTROL_FILE_PATH = "/tmp/block_picker_control.txt"
CONTROL_CHECK_INTERVAL_SEC = 0.2

# ============================================================
# 그리퍼 너비 설정 (mm)
# ============================================================
GRIPPER_OPEN    = 500      # 열기: 50mm (0.1mm 단위)
GRIPPER_PARTIAL = 200      # 중간: 25mm (0.1mm 단위)
GRIPPER_CLOSE   = 140      # 닫기: 14mm (0.1mm 단위)
GRIPPER_FORCE   = 400      # 파지력: 40N (0.1N 단위)

class SkipCurrentBrick(Exception):
    """사용자가 현재 brick을 건너뛰기로 선택했을 때 사용."""
    pass


# ============================================================
# 데이터 구조
# ============================================================
@dataclass
class Brick:
    index: int
    row: int
    col: int
    width: int
    height: int
    color: str = "NONE"
    z_layers: int = 1

    @classmethod
    def from_dict(cls, index: int, data: Dict[str, Any]) -> "Brick":
        col_key = next((k for k in ("col", "column") if k in data), None)
        if col_key is None:
            raise ValueError(f"Brick {index}: missing 'col' or 'column'")

        for key in ("row", "width", "height"):
            if key not in data:
                raise ValueError(f"Brick {index}: missing '{key}'")

        return cls(
            index=index,
            row=int(data["row"]),
            col=int(data[col_key]),
            width=int(data["width"]),
            height=int(data["height"]),
            color=str(data.get("color", "NONE")),
            z_layers=int(data.get("z_layers", 1)),
        )


# ============================================================
# 파일 로드 함수군
# ============================================================
def load_json(path: str) -> Dict[str, Any]:
    """
    JSON 파일을 읽어 dict로 반환

    Args:
        path: JSON 파일 절대 경로

    Returns:
        파싱된 dict

    Raises:
        FileNotFoundError: 파일이 존재하지 않는 경우
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: str) -> Dict[str, Any]:
    """
    YAML 파일을 읽어 dict로 반환

    Args:
        path: YAML 파일 절대 경로

    Returns:
        파싱된 dict

    Raises:
        FileNotFoundError: 파일이 존재하지 않는 경우
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"YAML file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_bricks(blocks_path: str) -> List[Brick]:
    """
    blocks.json에서 Brick 목록을 로드

    Args:
        blocks_path: blocks.json 절대 경로

    Returns:
        Brick 인스턴스 리스트 (index는 1부터 시작)

    Raises:
        ValueError: 'bricks' 키가 없거나 list가 아닌 경우
    """
    data = load_json(blocks_path)
    bricks_raw = data.get("bricks")
    if not isinstance(bricks_raw, list):
        raise ValueError("blocks.json must contain 'bricks' list")
    return [Brick.from_dict(i, item) for i, item in enumerate(bricks_raw, start=1)]


# ============================================================
# BoardMapper: 좌표 및 벡터 계산 보정
# ============================================================
_YAW_CFG_KEY = {
    (2, 3): "yaw_2x3_deg",
    (3, 2): "yaw_3x2_deg",
    (2, 2): "yaw_2x2_deg",
}


class BoardMapper:
    """캘리브레이션 3점 기반으로 보드 셀 인덱스를 로봇 베이스 좌표계 posx로 변환."""

    def __init__(self, calib: Dict[str, Any], logger) -> None:
        """
        캘리브레이션 YAML을 파싱해 보드 좌표계 벡터를 초기화

        Args:
            calib: board_calibration.yaml 파싱 결과
            logger: ROS2 Logger 인스턴스

        Note:
            row_vec/col_vec의 Z 성분은 캘리브레이션 측정 오차로 인한
            누적 Z 틀어짐을 방지하기 위해 강제로 0으로 설정한다.
        """
        self.calib = calib
        self.logger = logger

        grid = calib.get("grid", {})
        self.rows = int(grid.get("rows", 24))
        self.cols = int(grid.get("cols", 24))
        self.origin_mode = str(grid.get("origin_mode", "cell_center"))

        if self.origin_mode != "cell_center":
            raise ValueError("board_calibration.yaml의 grid.origin_mode를 'cell_center'로 설정하세요.")

        points = calib["calibration"]["points"]
        # 캘리브레이션 측정 3점: robot base 좌표계, mm 단위
        self.top_left_posx = np.array(points["top_left"]["posx"], dtype=float)
        self.bottom_left_posx = np.array(points["bottom_left"]["posx"], dtype=float)
        self.bottom_right_posx = np.array(points["bottom_right"]["posx"], dtype=float)

        self.top_left_xyz = self.top_left_posx[:3]      # robot base 좌표계, mm
        self.bottom_left_xyz = self.bottom_left_posx[:3]
        self.bottom_right_xyz = self.bottom_right_posx[:3]

        # 셀 간격 벡터 (mm/cell): 보드 XY 평면 기반
        self.row_vec = (self.bottom_left_xyz - self.top_left_xyz) / (self.rows - 1)
        self.col_vec = (self.bottom_right_xyz - self.bottom_left_xyz) / (self.cols - 1)

        # 캘리브레이션 측정 오차로 인한 Z 기울기 제거:
        # row/col이 커질수록 Z가 누적되어 삽입 깊이가 틀어지는 현상 방지
        _row_z = self.row_vec[2]
        _col_z = self.col_vec[2]
        self.row_vec[2] = 0.0
        self.col_vec[2] = 0.0
        self.logger.info(f"[Mapper] row_vec Z 강제 제거: {_row_z:+.6f} mm/cell")
        self.logger.info(f"[Mapper] col_vec Z 강제 제거: {_col_z:+.6f} mm/cell")

        self.base_abc = self.top_left_posx[3:].copy()
        # 블록 삽입 시 수직 자세를 보장하기 위해 A/B를 실측 수직 자세로 고정
        # C(yaw)는 블록 방향별로 달라지므로 측정값 그대로 유지
        self.base_abc[0] = INSERT_VERTICAL_A_DEG   # 수직 자세 A축 (deg)
        self.base_abc[1] = INSERT_VERTICAL_B_DEG   # 수직 자세 B축 (deg)

        self.yaw_cfg = calib.get("block_orientation", {})
        angle_cfg = calib.get("gripper_angle", {})
        # 그리퍼 미세 기울기 보정: 삽입 직전에만 C축에 적용 (deg)
        self.gripper_c_offset_deg: float = float(angle_cfg.get("c_offset_deg", DEFAULT_GRIPPER_C_OFFSET_DEG))

    def validate_brick(self, brick: Brick) -> None:
        """
        Brick이 보드 경계 내에 있는지 검증

        Raises:
            ValueError: row/col이 음수이거나 보드 범위를 초과하는 경우
        """
        if brick.row < 0 or brick.col < 0:
            raise ValueError(f"Brick {brick.index}: row/col cannot be negative")
        if brick.row + brick.height > self.rows or brick.col + brick.width > self.cols:
            raise ValueError(f"Brick {brick.index}: board boundary exceeded")

    def center_index(self, brick: Brick) -> Tuple[float, float]:
        """JSON row/col을 그대로 목표 셀 인덱스로 반환 (보드 전체 중심이 아닌 지정 셀 기준)."""
        return float(brick.row), float(brick.col)

    def yaw_offset_deg(self, brick: Brick) -> float:
        """
        블록 크기(width×height)에 따른 yaw 보정각 반환 (deg)

        Returns:
            YAML의 block_orientation에서 읽은 yaw 값, 미정의 크기는 0.0
        """
        key = _YAW_CFG_KEY.get((brick.width, brick.height))
        if key is None:
            return 0.0
        return float(self.yaw_cfg.get(key, 0.0))

    def apply_insert_offset(
        self,
        cell_xyz: np.ndarray,
        offset_x_mm: float,  # robot base X 방향 오프셋 (mm)
        offset_y_mm: float,  # robot base Y 방향 오프셋 (mm)
    ) -> np.ndarray:
        """
        셀 중심에서 그립 보정 오프셋을 적용한 TCP 목표 좌표를 반환

        Note:
            보드 방향 벡터가 아닌 로봇 베이스 좌표계 기준으로 직접 보정한다.
        """
        tcp_xyz = np.array(cell_xyz, dtype=float).copy()
        tcp_xyz[0] += offset_x_mm
        tcp_xyz[1] += offset_y_mm
        return tcp_xyz

    def brick_to_xyz(
        self,
        brick: Brick,
        offset_x_mm: float,  # robot base X 방향 그립 보정 (mm)
        offset_y_mm: float,  # robot base Y 방향 그립 보정 (mm)
    ) -> np.ndarray:
        """
        Brick의 셀 중심에 그립 오프셋을 적용한 xyz를 반환 (robot base 좌표계, mm)

        Returns:
            (3,) 배열, robot base 좌표계 mm 단위
        """
        self.validate_brick(brick)
        target_cell_row, target_cell_col = self.center_index(brick)
        cell_center_xyz = self.top_left_xyz + self.row_vec * target_cell_row + self.col_vec * target_cell_col
        return self.apply_insert_offset(cell_center_xyz, offset_x_mm, offset_y_mm)

    def cell_center_posx(self, brick: Brick, z_offset_mm: float = 0.0) -> List[float]:
        """
        그립 오프셋 미적용 셀 중심 posx를 반환 (접근 경로 첫 번째 지점용)

        Args:
            brick: 대상 Brick
            z_offset_mm: Z 방향 추가 오프셋 (mm)

        Returns:
            [x, y, z, rx, ry, rz] robot base 좌표계 (mm, deg)
        """
        self.validate_brick(brick)
        r, c = self.center_index(brick)
        xyz = self.top_left_xyz + self.row_vec * r + self.col_vec * c
        xyz[2] += z_offset_mm
        abc = self.base_abc.copy()
        abc[2] += self.yaw_offset_deg(brick)
        return [float(v) for v in [*xyz, *abc]]

    def brick_to_posx(
        self,
        brick: Brick,
        z_offset_mm: float,  # Z 방향 추가 오프셋 (mm)
        offset_x_mm: float,  # 그립 보정 X (mm)
        offset_y_mm: float,  # 그립 보정 Y (mm)
    ) -> List[float]:
        """
        그립 오프셋과 Z 오프셋을 적용한 최종 삽입 위치 posx를 반환

        Returns:
            [x, y, z, rx, ry, rz] robot base 좌표계 (mm, deg)
        """
        xyz = self.brick_to_xyz(brick, offset_x_mm, offset_y_mm)
        xyz[2] += z_offset_mm
        abc = self.base_abc.copy()
        abc[2] += self.yaw_offset_deg(brick)
        return [float(v) for v in [*xyz, *abc]]

    def print_info(self) -> None:
        """캘리브레이션 요약 정보를 stdout에 출력."""
        print("\n========== Board Calibration Info ==========")
        print(f"origin_mode       : {self.origin_mode}")
        print(f"top_left xyz      : {self.top_left_xyz}")
        print(f"row cell length   : {np.linalg.norm(self.row_vec):.3f} mm")
        print(f"col cell length   : {np.linalg.norm(self.col_vec):.3f} mm")
        print("===========================================\n")


# ============================================================
# Robot Controller
# ============================================================
class RobotController(Node):
    def __init__(self):
        super().__init__("m0609_block_picker")
        self._load_config()
        self.mapper = BoardMapper(self.calib, self.get_logger())
        self.mapper.print_info()
        self.gripper = self._create_gripper()

    def _load_config(self) -> None:
        """
        ROS2 파라미터 및 캘리브레이션 파일을 로드해 멤버 변수를 초기화

        처리 흐름:
        1. calibration_path / blocks_path 파라미터 선언 및 YAML 로드
        2. 단일 블럭 파라미터(row/col/width/height)가 있으면 단일 모드로 동작
        3. 없으면 blocks.json에서 전체 Brick 목록 로드
        4. motion/insert/column_correction/home 설정 파싱
        """
        pkg = get_package_share_directory(PACKAGE_NAME)

        self.declare_parameter("calibration_path", os.path.join(pkg, "config", "board_calibration.yaml"))
        self.declare_parameter("blocks_path", os.path.join(pkg, "config", "blocks.json"))
        self.declare_parameter("limit_count", 0)

        # 단일 블럭 모드: ros2 run 시 --ros-args -p row:=5 -p col:=3 ... 형태로 지정
        self.declare_parameter("row", -1)
        self.declare_parameter("col", -1)
        self.declare_parameter("width", -1)
        self.declare_parameter("height", -1)
        self.declare_parameter("z_layers", 1)

        calib_path = self.get_parameter("calibration_path").value
        self.calib = load_yaml(calib_path)
        self.get_logger().info(f"calibration_path: {calib_path}")

        row = int(self.get_parameter("row").value)
        col = int(self.get_parameter("col").value)
        width = int(self.get_parameter("width").value)
        height = int(self.get_parameter("height").value)
        z_layers = int(self.get_parameter("z_layers").value)

        if all(v >= 0 for v in (row, col, width, height)):
            self.get_logger().info(f"[단일 블럭 모드] row={row} col={col} width={width} height={height} layer={z_layers}")
            self.bricks = [Brick(index=1, row=row, col=col, width=width, height=height, z_layers=z_layers)]
        else:
            blocks_path = self.get_parameter("blocks_path").value
            limit_count = int(self.get_parameter("limit_count").value)
            self.get_logger().info(f"blocks_path: {blocks_path}")
            self.bricks = load_bricks(blocks_path)
            if limit_count > 0:
                self.bricks = self.bricks[:limit_count]

        motion = self.calib.get("motion", {})
        self.velocity = float(motion.get("velocity", 60))
        self.acceleration = float(motion.get("acceleration", 60))
        self.approach_z_mm = float(motion.get("approach_z_mm", 80.0))
        self.lift_z_mm = float(motion.get("lift_z_mm", 80.0))
        self.pick_z_offset_mm = float(motion.get("pick_z_offset_mm", 0.0))

        insert_cfg = self.calib.get("block_insert", {})
        self.insert_offset_x_mm = float(insert_cfg.get("offset_x_mm", DEFAULT_INSERT_OFFSET_X_MM))
        self.insert_offset_y_mm = float(insert_cfg.get("offset_y_mm", DEFAULT_INSERT_OFFSET_Y_MM))

        col_corr_cfg = self.calib.get("column_correction", {})
        self.column_correction_enabled = bool(col_corr_cfg.get("enabled", COLUMN_CORRECTION_ENABLED))
        self.column_correction_reference_col = int(col_corr_cfg.get("reference_col", COLUMN_CORRECTION_REFERENCE_COL))
        self.column_correction_x_mm = float(col_corr_cfg.get("x_mm", COLUMN_CORRECTION_X_MM))
        self.column_correction_y_mm = float(col_corr_cfg.get("y_mm", COLUMN_CORRECTION_Y_MM))
        self.column_correction_mode = str(col_corr_cfg.get("mode", COLUMN_CORRECTION_MODE))

        home_cfg = self.calib.get("home", {})
        self.home_posx = home_cfg.get("posx", HOME_POSX)
        self.home_posj = home_cfg.get("posj", HOME_POSJ)
        self.dry_run = bool(self.calib.get("dry_run", True))

        self.pause_requested = False
        self.stop_requested = False
        self.current_brick_context = None
        self._clear_control_file()

    def _create_gripper(self) -> Optional[RG]:
        gcfg = self.calib.get("gripper", {})
        name = gcfg.get("name", "rg2")
        ip = gcfg.get("toolchanger_ip", "192.168.1.1")
        port = gcfg.get("toolchanger_port", "502")
        self.get_logger().info(f"Gripper: name={name}, ip={ip}, port={port}")
        if self.dry_run:
            return None
        try:
            return RG(name, ip, port)
        except Exception as e:
            self.get_logger().warn(f"Gripper 연결 실패: {e}")
            return None

    # ============================================================
    # 비동기 파일 제어 인터페이스 (키 입력 무관 백그라운드 체크)
    # ============================================================
    def _clear_control_file(self) -> None:
        """제어 파일을 삭제해 이전 명령이 재처리되지 않도록 초기화."""
        try:
            if os.path.exists(CONTROL_FILE_PATH):
                os.remove(CONTROL_FILE_PATH)
        except Exception as e:
            self.get_logger().warn(f"Control file clear failed: {e}")

    def _consume_control_command(self) -> Optional[str]:
        """
        제어 파일에서 명령을 읽고 파일을 즉시 삭제 (단발성 소비)

        Returns:
            명령 문자열 또는 None (파일 없거나 내용 없을 때)

        Note:
            읽은 직후 파일을 삭제해 동일 명령이 중복 처리되는 것을 방지한다.
        """
        if not os.path.exists(CONTROL_FILE_PATH):
            return None
        try:
            with open(CONTROL_FILE_PATH, "r", encoding="utf-8") as f:
                cmd = f.read().strip().lower()
            self._clear_control_file()
            return cmd if cmd else None
        except Exception as e:
            self.get_logger().warn(f"Control file read failed: {e}")
            return None

    def _process_control_command(self) -> None:
        """
        제어 파일 명령(pause/resume/stop)을 처리해 상태 플래그를 갱신

        Note:
            각 movel/movej 전후에 호출되는 안전 정지 방식이며,
            motion 도중 강제 중단(emergency stop)은 아니다.
        """
        cmd = self._consume_control_command()
        if cmd is None:
            return
        if cmd in ["pause", "p"]:
            self.pause_requested = True
            self.get_logger().info("pause 요청 감지. 다음 동작 전 정지합니다.")
        elif cmd in ["resume", "r"]:
            self.pause_requested = False
            self.get_logger().info("resume 요청 감지. 실행 재개.")
        elif cmd in ["stop", "q", "quit"]:
            self.stop_requested = True
            raise KeyboardInterrupt

    def _wait_if_paused(self) -> None:
        """
        pause 상태라면 resume 또는 stop 명령이 들어올 때까지 폴링 대기

        Note:
            CONTROL_CHECK_INTERVAL_SEC 간격으로 제어 파일을 확인한다.
        """
        self._process_control_command()
        if not self.pause_requested:
            return
        self.get_logger().info(f"[PAUSED] 대기 중... 재개(echo resume > {CONTROL_FILE_PATH})")
        while self.pause_requested:
            time.sleep(CONTROL_CHECK_INTERVAL_SEC)
            self._process_control_command()

    def _check_stop_or_pause(self) -> None:
        """각 이동 명령 전/후에 호출해 pause/stop 상태를 확인하고 처리."""
        self._wait_if_paused()
        if self.stop_requested:
            raise KeyboardInterrupt

    # ============================================================
    # 오프셋 및 보정 필터
    # ============================================================
    def _set_z(self, posx: List[float], z_mm: float) -> List[float]:
        """posx의 Z값만 z_mm으로 교체해 반환 (robot base 좌표계, mm)."""
        p = list(posx)
        p[2] = z_mm
        return p

    def _column_extra_correction(self, brick: Brick) -> Tuple[float, float]:
        """
        특정 col 이상에서 발생하는 기구적 오차를 보정하는 추가 XY 오프셋 반환

        Returns:
            (dx_mm, dy_mm): robot base 좌표계 추가 보정값 (mm)
        """
        if not self.column_correction_enabled:
            return 0.0, 0.0
        if self.column_correction_mode == "equal":
            apply = (brick.col == self.column_correction_reference_col)
        else:
            apply = (brick.col >= self.column_correction_reference_col)
        if apply:
            return self.column_correction_x_mm, self.column_correction_y_mm
        return 0.0, 0.0

    def _apply_xy_extra_correction(
        self,
        posx: List[float],
        dx: float,  # robot base X 방향 추가 보정 (mm)
        dy: float,  # robot base Y 방향 추가 보정 (mm)
    ) -> List[float]:
        """posx의 XY에 추가 보정을 적용한 복사본을 반환."""
        p = list(posx)
        p[0] += dx
        p[1] += dy
        return p

    def _apply_gripper_angle_before_insert(self, posx: List[float]) -> List[float]:
        """
        삽입 직전 접근 높이에서만 그리퍼 C축 각도 보정을 적용

        Note:
            모든 이동에 적용하지 않고 삽입 직전에만 적용하는 이유:
            경로 이동 중 C축 틀어짐이 없도록 마지막 단계에서만 보정한다.
        """
        p = list(posx)
        p[5] += self.mapper.gripper_c_offset_deg
        return p

    # ============================================================
    # 로봇 기본 구동 프리미티브 함수군 (input 제거)
    # ============================================================
    def _move_l(self, posx: List[float], label: str) -> None:
        """
        직선 이동(movel) 실행 wrapper

        Args:
            posx: 목표 TCP 위치 [x,y,z,rx,ry,rz] (robot base 좌표계, mm/deg)
            label: 로그 식별용 레이블

        Note:
            dry_run=True이면 실제 이동 없이 로그만 출력한다.
            이동 전후로 pause/stop 상태를 확인한다.
        """
        self._check_stop_or_pause()
        self.get_logger().info(f"MOVE {label}: {posx}")
        if self.dry_run:
            return
        movel(posx, vel=self.velocity, acc=self.acceleration)
        mwait()
        self._check_stop_or_pause()

    def _move_j(self, joint: List[float], label: str) -> None:
        """
        관절 이동(movej) 실행 wrapper

        Args:
            joint: 목표 조인트 각도 6-DOF (deg)
            label: 로그 식별용 레이블
        """
        self._check_stop_or_pause()
        self.get_logger().info(f"MOVEJ {label}: {joint}")
        if self.dry_run:
            return
        movej(joint, vel=self.velocity, acc=self.acceleration)
        mwait()
        self._check_stop_or_pause()

    def _gripper_move(self, width_01mm: int, label: str) -> None:
        """
        그리퍼를 지정 너비로 이동하고 동작 완료까지 대기

        Args:
            width_01mm: 목표 너비 (0.1mm 단위). GRIPPER_OPEN / GRIPPER_CLOSE 상수 사용
            label: 로그 식별용 레이블
        """
        self.get_logger().info(f"[GRIPPER] {label} → {width_01mm * 0.1} mm")
        if self.dry_run or self.gripper is None:
            return
        self.gripper.move_gripper(width_01mm, GRIPPER_FORCE)
        time.sleep(1.5)  # RG2는 완전 닫힘→열림에 1~2초 필요, 0.5초면 열리기 전에 로봇이 올라감

    def _gripper_open(self) -> None:
        """그리퍼를 완전히 열고 동작 완료까지 대기 (GRIPPER_OPEN=500, 50mm)."""
        self._gripper_move(GRIPPER_OPEN, "Open")

    def _gripper_partial(self) -> None:
        """그리퍼를 중간 너비로 이동하고 동작 완료까지 대기 (GRIPPER_PARTIAL=200, 20mm)."""
        self._gripper_move(GRIPPER_PARTIAL, "Partial")

    def _gripper_close(self) -> None:
        """그리퍼를 완전히 닫고 동작 완료까지 대기 (GRIPPER_CLOSE=140, 14mm)."""
        self._gripper_move(GRIPPER_CLOSE, "Close")
    
    def _init_robot(self) -> None:
        """
        TCP/Tool 설정 후 홈 조인트 자세(home_posj)로 이동

        Note:
            Doosan 일부 펌웨어에서 AUTONOMOUS 상태에서 set_tool/set_tcp가 무시되므로
            반드시 MANUAL 모드로 전환 후 설정하고 AUTONOMOUS로 복귀한다.
        """
        self._check_stop_or_pause()
        if not self.dry_run:
            from DSR_ROBOT2 import set_tool, set_tcp, ROBOT_MODE_MANUAL, ROBOT_MODE_AUTONOMOUS, set_robot_mode
            set_robot_mode(ROBOT_MODE_MANUAL)
            time.sleep(0.5)  # 모드 전환 안정화 대기
            set_tool(ROBOT_TOOL)
            set_tcp(ROBOT_TCP)
            set_robot_mode(ROBOT_MODE_AUTONOMOUS)
            time.sleep(2.0)  # AUTONOMOUS 복귀 후 안정화 대기

        self._move_j(self.home_posj, "HOME_POSJ")

    def _try_ik_print(self, posx: List[float], label: str = "") -> None:
        """
        IK 해 존재 여부를 사전 검증해 도달 불가 좌표를 조기에 감지

        Note:
            설치 버전마다 ikin 시그니처가 다를 수 있으므로 실패 시 경고만 출력하고 진행한다.
        """
        if ikin is None:
            return
        try:
            result = ikin(posx, 0)
            self.get_logger().info(f"IK {label} sol_space=0: {result}")
        except Exception as e:
            self.get_logger().warn(f"IK {label}: ikin check skipped: {e}")

    # ============================================================
    # 비즈니스 시퀀스 제어 로직
    # ============================================================
    def _run_release_sequence(self, target_pos: List[float], lift_pos: List[float]) -> None:
        """
        블럭 삽입 후 그리퍼 해제 시퀀스 실행.

        처리 흐름:
        1. 그리퍼 50mm 열기 — 블럭에서 손가락 분리
        2. lift_pos로 상승 — 블럭 위로 후퇴
        3. 그리퍼 25mm로 좁히기 — 다음 집기 준비 자세
        4. 순응제어로 13.0mm 추가 하강 — 블럭을 핀에 완전히 눌러 끼움
        5. lift_pos로 최종 상승 — 작업 완료 후 후퇴

        Args:
            target_pos: 블럭 삽입 목표 좌표 (Z 기준으로 press 깊이 계산)
            lift_pos: 상승 복귀 좌표 (삽입 전후 안전 높이)

        Note:
            -13.0mm는 블럭 핀이 보드에 완전히 맞물리기 위한 실측 압입 깊이.
            순응제어(_compliance_down)를 사용하는 이유: 위치 오차가 있어도
            힘 기반으로 하강해 블럭 또는 보드 파손을 방지하기 위함.
        """
        # target_pos Z에서 13.0mm 더 내려간 압입 목표 좌표 계산
        release_down_pos = self._set_z(target_pos, target_pos[2] - 13.0)

        self._gripper_open()                        # 1. 그리퍼 50mm 열어 블럭 해제
        self._move_l(lift_pos, "release_lift_up")   # 2. 안전 높이로 상승
        self._gripper_partial()                     # 3. 그리퍼 25mm로 좁혀 다음 집기 준비
        self._compliance_down(release_down_pos)     # 4. 순응제어로 블럭을 핀에 완전 압입
        self._move_l(lift_pos, "release_lift_final") # 5. 최종 상승 후퇴


    def _compliance_down(self, pos: List[float]) -> None:
        """
        순응제어(task compliance)를 활성화한 상태로 하강 후 해제

        처리 흐름:
        1. task_compliance_ctrl로 순응제어 시작
        2. set_stiffnessx로 강성 설정 (Z축 유연, XY/회전 강성 유지)
        3. movel로 하강
        4. release_compliance_ctrl로 순응제어 해제

        Note:
            강성값 [500, 500, 100, 100, 100, 100]:
            - XY: 500 N/m (단단하게 유지)
            - Z:  100 N/m (유연하게 → 핀 접촉 시 과부하 방지)
            - 회전 3축: 100 N·m/rad
            dry_run 또는 함수 미지원 시 일반 movel로 대체한다.
        """
        if self.dry_run or task_compliance_ctrl is None:
            self.get_logger().warn("[COMPLIANCE] dry_run 또는 task_compliance_ctrl 미지원 → 일반 movel로 대체")
            self._move_l(pos, "release_down_again(no_compliance)")
            return

        self.get_logger().info("[COMPLIANCE] 순응제어 시작")
        try:
            ret = task_compliance_ctrl([500, 500, 100, 100, 100, 100])
            # DSR_ROBOT2는 실패 시 -1 또는 False 반환 (버전마다 다름)
            if ret is not None and ret < 0:
                self.get_logger().error(f"[COMPLIANCE] task_compliance_ctrl 실패 (ret={ret}) → 일반 movel로 대체")
                self._move_l(pos, "release_down_again(compliance_failed)")
                return
        except Exception as e:
            self.get_logger().error(f"[COMPLIANCE] task_compliance_ctrl 예외: {e} → 일반 movel로 대체")
            self._move_l(pos, "release_down_again(compliance_error)")
            return

        time.sleep(1.0)  # DSR 컨트롤러가 순응제어 모드를 실제 적용할 때까지 대기 (0.5s는 부족할 수 있음)
        try:
            self._move_l(pos, "release_down_again(compliance)")
        finally:
            try:
                release_compliance_ctrl()
            except Exception as e:
                self.get_logger().error(f"[COMPLIANCE] release_compliance_ctrl 예외: {e}")
            self.get_logger().info("[COMPLIANCE] 순응제어 해제")

    def _process_brick_motion(self, brick: Brick, current_insert_z: float) -> None:
        """
        단일 Brick에 대한 전체 삽입 모션 시퀀스 실행

        처리 흐름:
        1. 셀 중심 상공(cell_center_approach)으로 이동
        2. 그립 오프셋 적용 위치 상공(diagonal_offset)으로 이동
        3. 그리퍼 C축 보정(rotate_before_C_offset) 후 접근
        4. insert Z로 하강
        5. 그리퍼 해제 시퀀스 실행

        Args:
            brick: 삽입할 Brick 정보
            current_insert_z: 현재 층의 삽입 Z 좌표 (robot base 좌표계, mm)
        """
        cell_center_pos = self.mapper.cell_center_posx(brick, z_offset_mm=self.pick_z_offset_mm)

        if brick.width == 2 and brick.height == 3:
            offset_x, offset_y = INSERT_2X3_OFFSET_X_MM, INSERT_2X3_OFFSET_Y_MM
        else:
            offset_x, offset_y = self.insert_offset_x_mm, self.insert_offset_y_mm

        raw_target_pos = self.mapper.brick_to_posx(
            brick, z_offset_mm=self.pick_z_offset_mm, offset_x_mm=offset_x, offset_y_mm=offset_y
        )

        dx, dy = self._column_extra_correction(brick)
        raw_target_pos = self._apply_xy_extra_correction(raw_target_pos, dx, dy)

        approach_z = current_insert_z + self.approach_z_mm
        cell_approach_pos = self._set_z(cell_center_pos, approach_z)
        diagonal_approach_pos = self._set_z(raw_target_pos, approach_z)
        rotate_before_insert_pos = self._apply_gripper_angle_before_insert(diagonal_approach_pos)
        target_pos = self._set_z(rotate_before_insert_pos, current_insert_z)
        lift_pos = self._set_z(rotate_before_insert_pos, current_insert_z + self.lift_z_mm)

        self.current_brick_context = f"idx:{brick.index} | pos:{brick.row},{brick.col} | layer:{brick.z_layers}"
        self._try_ik_print(target_pos, label=f"brick {brick.index}")

        # 무인 완전 자동 궤적 구동
        self._move_l(cell_approach_pos, "cell_center_approach")
        self._move_l(diagonal_approach_pos, "diagonal_offset")
        self._move_l(rotate_before_insert_pos, "rotate_before_C_offset")
        self._move_l(target_pos, f"target_down_layer_{brick.z_layers}")
        self._run_release_sequence(target_pos, lift_pos)

    def run(self) -> None:
        """
        전체 Brick 목록을 층(z_layers)별로 순차 삽입 실행

        처리 흐름:
        1. 홈 위치로 초기 이동
        2. z_layers 오름차순으로 층별 그룹 처리
        3. 각 층의 Brick마다 _process_brick_motion 호출
        4. 완료 후 홈 위치로 복귀
        """
        self.get_logger().info(f"Run Start — dry_run={self.dry_run}, bricks={len(self.bricks)}")
        self._init_robot()

        layers = sorted(set(b.z_layers for b in self.bricks))

        for layer in layers:
            layer_bricks = [b for b in self.bricks if b.z_layers == layer]
            self.get_logger().info(f"========== {layer}층 공정 개시: {len(layer_bricks)}개 블록 ==========")

            current_insert_z = LAYER_INSERT_Z.get(layer, LAYER_INSERT_Z[1])  # mm, robot base Z 좌표

            for brick in layer_bricks:
                self._check_stop_or_pause()
                self._process_brick_motion(brick, current_insert_z)

                self.current_brick_context = None

        self.get_logger().info("모든 블록 공정 완료. 홈 위치로 복귀합니다.")
        self._init_robot()


# ============================================================
# Main 진입점
# ============================================================
def main(args: Optional[list] = None) -> None:
    """
    ROS2 노드 초기화 후 RobotController를 실행하고 종료 처리

    처리 흐름:
    1. rclpy 초기화 및 DSR 노드 생성 (DR_init 연결 후 DSR_ROBOT2 import 가능)
    2. DSR_ROBOT2 함수 import (ikin 등 선택 함수는 없어도 무시)
    3. RobotController 생성 및 run() 실행
    4. 예외 발생 시 노드 정리 및 shutdown
    """
    global movej, movel, mwait, ikin, task_compliance_ctrl, release_compliance_ctrl, set_stiffnessx

    rclpy.init(args=args)
    # DR_init에 노드를 연결해야 DSR_ROBOT2 내부 ROS 통신이 정상 동작한다
    dsr_node = rclpy.create_node("place_block", namespace=ROBOT_ID)
    DR_init.__dsr__node = dsr_node

    # DSR_ROBOT2 서비스 콜백 처리용 spin 스레드 (없으면 set_robot_mode 등이 블로킹됨)
    _executor = SingleThreadedExecutor()
    _executor.add_node(dsr_node)
    _spin_stop = threading.Event()

    def _spin_loop():
        while not _spin_stop.is_set():
            _executor.spin_once(timeout_sec=0.05)

    _spin_thread = threading.Thread(target=_spin_loop, daemon=True)
    _spin_thread.start()

    try:
        from DSR_ROBOT2 import movej, movel, mwait  # type: ignore
        import DSR_ROBOT2 as _dsr  # type: ignore
        # 설치 버전에 따라 일부 함수가 없을 수 있으므로 getattr로 방어적으로 가져온다
        ikin = getattr(_dsr, "ikin", None)
        task_compliance_ctrl    = getattr(_dsr, "task_compliance_ctrl", None)
        release_compliance_ctrl = getattr(_dsr, "release_compliance_ctrl", None)
        set_stiffnessx          = getattr(_dsr, "set_stiffnessx", None)
    except ImportError as e:
        print(f"[ERROR] Failed to import DSR_ROBOT2: {e}")
        os._exit(1)

    exit_code = 0
    node = None
    try:
        node = RobotController()
        node.run()
    except KeyboardInterrupt:
        print("\n[STOP] 사용자에 의한 강제 종료 (KeyboardInterrupt)")
        exit_code = 1
    except Exception as e:
        print(f"\n[ERROR] 예외 발생: {e}")
        exit_code = 1

    # spin 스레드 중단 후 즉시 종료 (destroy_node 시 SIGABRT 방지)
    _spin_stop.set()
    _spin_thread.join(timeout=3)
    try:
        if node is not None:
            node.gripper.close_connection()
    except Exception:
        pass
    os._exit(exit_code)


if __name__ == "__main__":
    main()