import logging
import os
import random
import subprocess
import threading
import time

import numpy as np

from config import ROS_SOURCE, SCRIPT_DIR, PICK_READY_SCRIPT, MOVING_PICK_SCRIPT
from ros.cam_state import cam_state, cam_lock
from state.robot_state import robot_state, state_lock
from process.process_manager import pick_proc_manager
from utils.geometry import compute_pick_target


log = logging.getLogger(__name__)


def do_move_pick_place_sync(timeout: int = 60) -> bool:
    """
    [자동화 전용] pick 준비 위치로 로봇을 이동시키는 ROS 노드를 동기 실행한다.

    automation_service에서 YOLO 탐지 전 준비 이동 단계로 사용.

    처리 흐름:
    1. subprocess로 move_pick_place 노드 기동
    2. timeout 내 종료 대기
    3. returncode 0이면 True 반환

    Args:
        timeout: 프로세스 최대 대기 시간 (초). 기본 60초.

    Returns:
        True: 정상 종료 (returncode=0)
        False: timeout 또는 비정상 종료

    Note:
        blocking 호출 — 호출 스레드가 timeout 동안 점유됨.
        60초는 가장 먼 준비 위치 이동 경로의 실측 상한값 기준.
    """
    env = os.environ.copy()
    env.setdefault("DISPLAY", ":0")

    proc = subprocess.Popen(
        ["bash", "-c", f"{ROS_SOURCE} && ros2 run pick_block move_pick_place"],
        cwd=SCRIPT_DIR,
        env=env,
    )
    pick_proc_manager.add(proc)
    log.info("[MOVE_PICK_PLACE] PID=%d 시작", proc.pid)

    try:
        proc.wait(timeout=timeout)
        log.info("[MOVE_PICK_PLACE] PID=%d 종료 returncode=%d", proc.pid, proc.returncode)
        return proc.returncode == 0
    except subprocess.TimeoutExpired:
        proc.kill()
        log.error("[MOVE_PICK_PLACE] PID=%d timeout(%ds) → 강제 종료", proc.pid, timeout)
        return False


def do_pick_sync(color: str, shape: str, timeout: int = 120) -> bool:
    """
    [자동화 전용] 탐지된 블럭 중 조건에 맞는 하나를 선택해 pick 스크립트를 동기 실행한다.

    automation_service에서 pick 성공 여부를 확인 후 place 진행 여부를 결정하기 위해 사용.

    처리 흐름:
    1. cam_state 스냅샷 (락 최소 구간)
    2. color/shape 조건 필터링
    3. 후보 중 랜덤 선택 + 3/4 지점 좌표 보정
    4. depth 픽셀 값 읽기 (0이면 3×3 주변 중앙값으로 대체)
    5. pick 스크립트에 픽셀 좌표·깊이·각도·카메라 파라미터 전달

    Args:
        color: 대상 색상 ("red"/"blue"/"yellow"/"all")
        shape: 대상 형태 ("2x2"/"2x3"/"all")
        timeout: 프로세스 최대 대기 시간 (초). 기본 120초.

    Returns:
        True: 정상 종료 (returncode=0)
        False: 후보 없음 / 깊이 없음 / timeout / 비정상 종료

    Note:
        blocking 호출.
        depth=0인 픽셀은 카메라 사각이나 반사로 미측정된 경우이므로
        주변 3×3 유효값의 중앙값으로 대체한다.
    """
    env = os.environ.copy()
    env.setdefault("DISPLAY", ":0")

    with cam_lock:
        detections: list[dict]    = list(cam_state["detections"])
        depth_frame: np.ndarray   = cam_state["depth_frame"]
        cam_info: dict | None     = cam_state["cam_info"]

    if depth_frame is None or cam_info is None:
        log.error("[PICK_SYNC] 카메라 정보 미수신")
        return False

    candidates: list[dict] = [
        d for d in detections
        if (color == "all" or d["color"] == color)
        and (shape == "all" or d["shape"] == shape)
    ]
    if not candidates:
        log.warning("[PICK_SYNC] 조건(%s/%s)에 맞는 블럭 없음 (탐지 %d개)", color, shape, len(detections))
        return False

    det: dict          = random.choice(candidates)
    cx: int
    cy: int
    cx, cy             = compute_pick_target(det)  # 픽셀 좌표
    angle_deg: float   = det["angle_deg"]           # deg

    h_img, w_img = depth_frame.shape[:2]
    cy_c: int = max(0, min(cy, h_img - 1))
    cx_c: int = max(0, min(cx, w_img - 1))
    cz: float = float(depth_frame[cy_c, cx_c])     # mm (uint16 depth)
    if cz == 0:
        # depth=0은 센서 사각이나 반사로 미측정된 픽셀 → 주변 유효값 중앙값으로 보완
        region = depth_frame[max(0, cy_c-2):cy_c+3, max(0, cx_c-2):cx_c+3]
        valid  = region[region > 0]
        cz     = float(np.median(valid)) if len(valid) > 0 else 0.0
    if cz == 0:
        log.error("[PICK_SYNC] depth=0 cx=%d cy=%d", cx, cy)
        return False

    log.info("[PICK_SYNC] 선택: color=%s shape=%s cx=%d cy=%d cz=%.1f angle=%.2f",
             det["color"], det["shape"], cx, cy, cz, angle_deg)

    # pick 스크립트에 픽셀 좌표(px)·깊이(mm)·각도(deg)·카메라 내부 파라미터(px) 전달
    proc = subprocess.Popen(
        ["bash", "-c",
         f"{ROS_SOURCE} && python3 {MOVING_PICK_SCRIPT} "
         f"{cx} {cy} {cz} {angle_deg} "
         f"{cam_info['fx']} {cam_info['fy']} {cam_info['ppx']} {cam_info['ppy']}"],
        cwd=SCRIPT_DIR,
        env=env,
        stderr=subprocess.PIPE,
    )
    pick_proc_manager.add(proc)
    log.info("[PICK_SYNC] PID=%d 시작", proc.pid)

    try:
        _, stderr_data = proc.communicate(timeout=timeout)
        log.info("[PICK_SYNC] PID=%d 종료 returncode=%d", proc.pid, proc.returncode)
        if proc.returncode != 0 and stderr_data:
            log.error("[PICK_SYNC] stderr:\n%s", stderr_data.decode(errors="replace").strip())
        return proc.returncode == 0
    except subprocess.TimeoutExpired:
        proc.kill()
        log.error("[PICK_SYNC] PID=%d timeout(%ds) → 강제 종료", proc.pid, timeout)
        return False


def do_pick_async(filter_color: str, filter_shape: str) -> None:
    """
    [버튼 전용] pick 전체 파이프라인(준비 이동 → YOLO 탐지 → pick 실행)을 비동기로 기동한다.

    /api/action/pick 버튼에서 호출. API가 즉시 응답해야 하므로 비동기 실행.
    성공/실패는 robot_state["status"]로 확인.

    처리 흐름:
    1. pick 준비 위치 이동 (subprocess, blocking 최대 60초)
    2. YOLO 활성화 + 2초 안정화 대기
    3. 탐지 스냅샷 후 YOLO 즉시 비활성화
    4. 조건 필터링 + 랜덤 선택 + 3/4 지점 보정
    5. depth 읽기 (0이면 주변 중앙값 대체)
    6. pick 스크립트 실행 (blocking 최대 120초)

    Args:
        filter_color: 대상 색상 ("red"/"blue"/"yellow"/"all")
        filter_shape: 대상 형태 ("2x2"/"2x3"/"all")

    Note:
        daemon=True 스레드로 실행 — Flask 메인 프로세스 종료 시 자동 회수됨.
        각 subprocess에 timeout을 설정해 hang 발생 시 자동 강제 종료.
    """
    def _run() -> None:
        env = os.environ.copy()
        env.setdefault("DISPLAY", ":0")

        # 1. 집기 준비 위치로 이동
        with state_lock:
            robot_state["status"] = "집기 준비 위치로 이동 중…"
        log.info("[PICK] moving_pick_place.py 실행")
        ready_proc = subprocess.Popen(
            ["bash", "-c", f"{ROS_SOURCE} && python3 {PICK_READY_SCRIPT}"],
            cwd=SCRIPT_DIR, env=env,
        )
        log.info("[PICK] 준비 이동 PID=%d", ready_proc.pid)
        try:
            # 60초: 준비 위치 이동 실측 상한. 초과 시 hang으로 간주해 강제 종료
            ready_proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            ready_proc.kill()
            log.error("[PICK] 준비 이동 timeout → 강제 종료")
            with state_lock:
                robot_state["status"] = "집기 실패: 준비 이동 timeout"
                robot_state["action"] = None
            return
        log.info("[PICK] 준비 이동 완료 returncode=%d", ready_proc.returncode)

        # 2. YOLO 활성화 + 안정화 대기
        with cam_lock:
            cam_state["yolo_on"]    = True
            cam_state["detections"] = []
        with state_lock:
            robot_state["status"] = "블럭 탐지 중…"
        log.info("[PICK] YOLO 활성화, 2초 안정화 대기")
        # 카메라 자동 노출 수렴에 2초 필요
        time.sleep(2)

        # 3. 탐지 결과 스냅샷 후 YOLO 즉시 끄기
        with cam_lock:
            detections: list[dict]  = list(cam_state["detections"])
            depth_frame: np.ndarray = cam_state["depth_frame"]
            cam_info: dict | None   = cam_state["cam_info"]
            cam_state["yolo_on"]    = False
            cam_state["detections"] = []
        log.info("[PICK] YOLO 비활성화, 탐지 결과 %d개 스냅샷", len(detections))

        if depth_frame is None or cam_info is None:
            log.error("[PICK] 카메라 정보 미수신")
            with state_lock:
                robot_state["status"] = "집기 실패: 카메라 미준비"
                robot_state["action"] = None
            return

        candidates: list[dict] = [
            d for d in detections
            if (filter_color == "all" or d["color"] == filter_color)
            and (filter_shape == "all" or d["shape"] == filter_shape)
        ]
        if not candidates:
            cond = f"{filter_color} / {filter_shape}"
            log.warning("[PICK] 조건(%s)에 맞는 블럭 없음 (탐지 %d개)", cond, len(detections))
            with state_lock:
                robot_state["status"] = f"집기 실패: {cond} 블럭 없음"
                robot_state["action"] = None
            return

        # 4. 랜덤 선택 + 3/4 보정 + depth 읽기
        det: dict        = random.choice(candidates)
        cx: int
        cy: int
        cx, cy           = compute_pick_target(det)  # 픽셀 좌표
        angle_deg: float = det["angle_deg"]           # deg

        h_img, w_img = depth_frame.shape[:2]
        cy_c: int = max(0, min(cy, h_img - 1))
        cx_c: int = max(0, min(cx, w_img - 1))
        cz: float = float(depth_frame[cy_c, cx_c])   # mm (uint16 depth)
        if cz == 0:
            # depth=0은 반사·사각으로 미측정 → 주변 유효값 중앙값으로 보완
            region = depth_frame[max(0, cy_c-2):cy_c+3, max(0, cx_c-2):cx_c+3]
            valid  = region[region > 0]
            cz     = float(np.median(valid)) if len(valid) > 0 else 0.0
        if cz == 0:
            log.error("[PICK] depth=0 cx=%d cy=%d", cx, cy)
            with state_lock:
                robot_state["status"] = "집기 실패: depth 값 없음"
                robot_state["action"] = None
            return

        log.info("[PICK] 선택: color=%s shape=%s cx=%d cy=%d cz=%.1f angle=%.2f (후보 %d개 중 랜덤)",
                 det["color"], det["shape"], cx, cy, cz, angle_deg, len(candidates))

        # 5. pick 스크립트 실행
        with state_lock:
            robot_state["status"] = "블럭 집기 수행 중…"

        # cx, cy: 픽셀 좌표 | cz: mm | angle_deg: deg | fx,fy,ppx,ppy: 픽셀
        pick_proc = subprocess.Popen(
            ["bash", "-c",
             f"{ROS_SOURCE} && python3 {MOVING_PICK_SCRIPT} "
             f"{cx} {cy} {cz} {angle_deg} "
             f"{cam_info['fx']} {cam_info['fy']} {cam_info['ppx']} {cam_info['ppy']}"],
            cwd=SCRIPT_DIR, env=env,
        )
        log.info("[PICK] 프로세스 시작 PID=%d", pick_proc.pid)
        pick_proc_manager.add(pick_proc)
        try:
            # 120초: pick 모션 실측 상한. 초과 시 hang으로 간주해 강제 종료
            pick_proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            pick_proc.kill()
            log.error("[PICK] PID=%d pick timeout → 강제 종료", pick_proc.pid)
            with state_lock:
                robot_state["status"] = "집기 실패: timeout"
                robot_state["action"] = None
            return
        log.info("[PICK] 완료 PID=%d returncode=%d", pick_proc.pid, pick_proc.returncode)

        with state_lock:
            robot_state["status"] = "블럭 집기 완료"
            robot_state["action"] = None
        # 집기 완료 메시지를 UI에 잠깐 표시 후 대기 상태로 복귀
        time.sleep(2)
        with state_lock:
            if robot_state["status"] == "블럭 집기 완료":
                robot_state["status"] = "대기 중"

    # daemon=True: 메인 프로세스 종료 시 pick 스레드도 자동 회수
    threading.Thread(target=_run, daemon=True).start()
