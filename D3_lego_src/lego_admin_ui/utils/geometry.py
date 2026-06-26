import logging
import numpy as np

log = logging.getLogger(__name__)


def read_depth_with_fallback(depth_frame: np.ndarray, cx: int, cy: int) -> float:
    """
    depth_frame의 (cx, cy) 위치에서 깊이값(mm)을 반환한다.

    해당 픽셀이 depth=0(센서 사각·반사)이면 3×3 주변 유효값의 중앙값으로 대체.

    Returns:
        깊이값 (mm). 주변도 모두 0이면 0.0 반환.
    """
    h, w = depth_frame.shape[:2]
    cy_c = max(0, min(cy, h - 1))
    cx_c = max(0, min(cx, w - 1))
    cz = float(depth_frame[cy_c, cx_c])
    if cz == 0:
        region = depth_frame[max(0, cy_c - 2):cy_c + 3, max(0, cx_c - 2):cx_c + 3]
        valid = region[region > 0]
        cz = float(np.median(valid)) if len(valid) > 0 else 0.0
    return cz


def compute_pick_target(det: dict) -> tuple[int, int]:
    """
    YOLO OBB 탐지 결과로부터 실제 파지 목표 픽셀 좌표를 계산한다.

    처리 흐름:
    1. shape != "2x3"이거나 pts 없으면 중심 좌표 그대로 반환
    2. 4개 꼭짓점에서 긴 변 벡터 추출
    3. 중심에서 긴 변 방향으로 25% 이동한 지점 반환

    Args:
        det: YOLO 탐지 딕셔너리 — 최소 {cx, cy, shape, pts} 포함

    Returns:
        (cx, cy): 파지 목표 픽셀 좌표 (이미지 픽셀 좌표계)
        파지 보정 불가 시 원래 (cx, cy) 반환

    Note:
        2x3 블럭은 중심이 아닌 3/4 지점(긴 변 기준 25% 이동)을 집어야
        그리퍼가 블럭을 안정적으로 파지할 수 있음.
        2x2 또는 pts 없으면 중심 그대로 사용.
    """
    if "pts" not in det or det["shape"] != "2x3":
        return det["cx"], det["cy"]

    try:
        pts: np.ndarray = np.array(det["pts"], dtype=float)

        # OBB 꼭짓점 평균 → 중심 좌표 (픽셀)
        cx_center: float = np.mean(pts[:, 0])
        cy_center: float = np.mean(pts[:, 1])

        # 더 긴 변을 기준 벡터로 선택 — 긴 변이 블럭의 장축 방향
        edge01: np.ndarray = pts[1] - pts[0]
        edge12: np.ndarray = pts[2] - pts[1]
        edge_for_angle: np.ndarray = (
            edge01 if np.linalg.norm(edge01) >= np.linalg.norm(edge12) else edge12
        )

        # 중심에서 장축 방향으로 25% 이동 → 3/4 파지 지점 (픽셀)
        cx_target: float = cx_center - (edge_for_angle[0] * 0.25)
        cy_target: float = cy_center - (edge_for_angle[1] * 0.25)
        log.debug("[3/4 보정] center=(%.1f,%.1f) → target=(%.1f,%.1f)",
                  cx_center, cy_center, cx_target, cy_target)
        return int(cx_target), int(cy_target)
    except Exception as e:
        log.warning("[3/4 보정] 계산 실패, 중심 사용: %s", e)
        return det["cx"], det["cy"]
