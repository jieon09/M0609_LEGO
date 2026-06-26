"""
object_segment.py  v3
─────────────────────
로컬 이미지 → 종이 검출 → 이진화 → 자동 객체 분할 → 저장

변경사항 v3:
  - 테두리 5% 이내 컴포넌트 → 배경 노이즈로 제거
  - 최소 면적 0.0002 → 스마일 눈도 인식
  - 타임스탬프 서브폴더로 저장 → 이전 결과 유지

사용법:
  python3 object_segment.py --image ~/Downloads/drawing.png
  python3 object_segment.py --image ~/Downloads/drawing.png --merge-dist 150
  python3 object_segment.py --image ~/Downloads/drawing.png --no-paper
"""

import argparse
import os
import sys
from typing import Optional
import cv2
import numpy as np
from datetime import datetime


# A4 용지 기준 처리 해상도 (픽셀). 가로형 A4 비율 유지
A4_W: int = 840
A4_H: int = 595

# ── 종이 검출 + 원근 변환 ─────────────────────────────────────────────
def order_points(pts: np.ndarray) -> np.ndarray:
    """4개의 점을 [좌상, 우상, 우하, 좌하] 순으로 정렬.

    Args:
        pts: 4x2 float32 좌표 배열 (이미지 픽셀 좌표)

    Returns:
        정렬된 4x2 float32 배열
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def detect_paper(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """이미지에서 A4 종이 영역을 검출하고 원근 변환 적용.

    처리 흐름:
    1. Otsu 이진화로 가장 큰 윤곽선 검출
    2. 꼭짓점 4개이면 A4 크기(840×595px)로 원근 변환, 아니면 리사이즈

    Args:
        img: BGR 원본 이미지

    Returns:
        (변환된 이미지, 시각화용 이미지) 튜플
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    paper_vis = img.copy()

    if not contours:
        print("[종이 검출] 없음 → 리사이즈만 적용")
        return cv2.resize(img, (A4_W, A4_H)), paper_vis

    largest = max(contours, key=cv2.contourArea)
    peri    = cv2.arcLength(largest, True)
    approx  = cv2.approxPolyDP(largest, 0.02 * peri, True)

    cv2.drawContours(paper_vis, [approx], -1, (0, 255, 0), 2)
    for pt in approx.reshape(-1, 2):
        cv2.circle(paper_vis, tuple(pt.astype(int)), 8, (0, 0, 255), -1)

    if len(approx) == 4:
        pts  = approx.reshape(4, 2).astype(np.float32)
        dst  = np.array([[0,0],[A4_W-1,0],[A4_W-1,A4_H-1],[0,A4_H-1]],
                        dtype=np.float32)
        M = cv2.getPerspectiveTransform(order_points(pts), dst)
        warped = cv2.warpPerspective(img, M, (A4_W, A4_H))
        print(f"[종이 검출] 성공 → {A4_W}×{A4_H} 원근 변환")
        return warped, paper_vis
    else:
        # 4점 미만이면 완전한 사각형 아님 → 단순 리사이즈로 폴백
        print(f"[종이 검출] 꼭짓점 {len(approx)}개 → 리사이즈만 적용")
        return cv2.resize(img, (A4_W, A4_H)), paper_vis


# ── 이진화 ────────────────────────────────────────────────────────────
def binarize(img: np.ndarray) -> np.ndarray:
    """BGR 이미지를 이진화하여 그림 선을 흰색으로 추출.

    처리 흐름:
    1. HSV V채널 추출 → CLAHE 적용
    2. Gaussian Blur → Adaptive Threshold
    3. Morphology Close → Dilate (선 두께 확대)

    Args:
        img: BGR 이미지 (A4_W × A4_H)

    Returns:
        이진 맵 (HxW, 0 또는 255)

    Note:
        Dilate iterations=3 — 가는 선도 연결 컴포넌트로 병합되도록 두께 확대
    """
    hsv     = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(hsv)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v       = clahe.apply(v)
    blurred = cv2.GaussianBlur(v, (11, 11), 0)
    binary  = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=21, C=15
    )
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    binary = cv2.dilate(binary, kernel, iterations=3)
    return binary


# ── 자동 객체 분할 ────────────────────────────────────────────────────
def auto_segment(
    binary: np.ndarray, merge_dist: int
) -> tuple[Optional[np.ndarray], int]:
    """연결 컴포넌트 분석 + Union-Find로 가까운 컴포넌트를 객체 단위로 병합.

    처리 흐름:
    1. 8-connectivity 연결 컴포넌트 레이블링
    2. 면적 필터 (전체의 0.02% 미만 제거)
    3. 테두리 5% 이내 컴포넌트 → 배경 노이즈로 제거
    4. Union-Find로 merge_dist 이내 컴포넌트 병합

    Args:
        binary: 이진 맵 (A4_H × A4_W)
        merge_dist: 컴포넌트 중심 간 거리 임계값 (픽셀). 이하면 같은 객체로 병합

    Returns:
        (result_map, n_objects) 튜플
        유효 컴포넌트가 없으면 (None, 0)
    """
    num_labels, label_map, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    # 노이즈 제거 기준 1: 전체 면적의 0.02% 미만 → 스마일 눈 크기까지는 포함 가능
    min_area = (A4_W * A4_H) * 0.0002  # 약 100px²

    # 노이즈 제거 기준 2: 테두리 5% 이내 bounding box — 스캔 경계 잡음 제거
    bx_min = A4_W * 0.05
    bx_max = A4_W * 0.95
    by_min = A4_H * 0.05
    by_max = A4_H * 0.95

    valid = []
    noise_count = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area < min_area:
            noise_count += 1
            continue

        # bounding box (이미지 픽셀 좌표)
        bx = stats[i, cv2.CC_STAT_LEFT]
        by = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]

        # 테두리 필터: bounding box가 테두리 영역에 걸쳐있으면 제거
        if (bx < bx_min or by < by_min or
                bx + bw > bx_max or by + bh > by_max):
            noise_count += 1
            print(f"  [노이즈] 컴포넌트 {i} 테두리 근처 제거 "
                  f"(x:{bx} y:{by} w:{bw} h:{bh} area:{area})")
            continue

        cx, cy = centroids[i]
        valid.append((i, cx, cy, area))

    print(f"[컴포넌트] 총 {num_labels-1}개 → "
          f"노이즈 {noise_count}개 제거 → 유효 {len(valid)}개")

    if not valid:
        print("[오류] 유효한 컴포넌트 없음")
        return None, 0

    # Union-Find: 중심 간 유클리디안 거리 < merge_dist 이면 같은 그룹으로 병합
    n      = len(valid)
    parent = list(range(n))

    def find(x: int) -> int:
        """경로 압축(path compression) 적용 find."""
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        pa, pb = find(a), find(b)
        if pa != pb:
            parent[pa] = pb

    for i in range(n):
        for j in range(i + 1, n):
            cx1, cy1 = valid[i][1], valid[i][2]
            cx2, cy2 = valid[j][1], valid[j][2]
            dist = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
            if dist < merge_dist:
                union(i, j)

    roots    = {}
    group_id = 0
    for i in range(n):
        root = find(i)
        if root not in roots:
            roots[root] = group_id
            group_id   += 1

    n_objects      = group_id
    label_to_group = {}
    for i, (comp_label, cx, cy, area) in enumerate(valid):
        label_to_group[comp_label] = roots[find(i)] + 1

    result_map = np.zeros_like(label_map, dtype=np.int32)
    for comp_label, gid in label_to_group.items():
        result_map[label_map == comp_label] = gid

    print(f"[병합] merge_dist={merge_dist}px → {n_objects}개 객체")
    return result_map, n_objects


# ── 객체별 크롭 + 저장 ────────────────────────────────────────────────
def extract_and_save(
    warped: np.ndarray,
    result_map: np.ndarray,
    n_objects: int,
    output_dir: str,
    base_name: str,
) -> list[str]:
    """객체별로 흰 배경에 크롭하여 JPG 저장.

    Args:
        warped: A4 크기 BGR 이미지 (이미지 픽셀 좌표)
        result_map: 객체 레이블 맵 (값 = 객체 ID, 0은 배경)
        n_objects: 총 객체 수
        output_dir: 저장 폴더 경로
        base_name: 파일명 접두사 (현재 미사용, 향후 확장용)

    Returns:
        저장된 파일 경로 목록
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths: list[str] = []
    padding = 20  # 객체 bounding box 주변 여백 (픽셀)

    for obj_id in range(1, n_objects + 1):
        mask   = (result_map == obj_id).astype(np.uint8) * 255
        coords = cv2.findNonZero(mask)
        if coords is None:
            continue

        x, y, w, h = cv2.boundingRect(coords)
        x1 = max(0, x - padding);    y1 = max(0, y - padding)
        x2 = min(A4_W, x+w+padding); y2 = min(A4_H, y+h+padding)

        white_bg   = np.ones((y2-y1, x2-x1, 3), dtype=np.uint8) * 255
        obj_region = warped[y1:y2, x1:x2]
        mask_crop  = mask[y1:y2, x1:x2]
        mask_3ch   = cv2.cvtColor(mask_crop, cv2.COLOR_GRAY2BGR)
        # 객체 마스크 영역만 원본 픽셀 사용, 나머지는 흰색 배경으로 대체
        result     = np.where(mask_3ch > 0, obj_region, white_bg)

        save_path = os.path.join(output_dir, f"obj{obj_id}.jpg")
        cv2.imwrite(save_path, result)
        saved_paths.append(save_path)
        print(f"[저장] 객체 {obj_id}: {save_path} ({x2-x1}×{y2-y1}px)")

    return saved_paths


# ── 시각화 ────────────────────────────────────────────────────────────
def visualize(
    orig: np.ndarray,
    paper_vis: np.ndarray,
    warped: np.ndarray,
    binary: np.ndarray,
    result_map: np.ndarray,
    n_objects: int,
    saved_paths: list[str],
) -> None:
    TARGET_H = 300
    COLORS   = [
        (0,0,220),(0,180,0),(220,100,0),
        (0,200,220),(180,0,180),(0,150,150),(100,100,220)
    ]

    seg_vis = np.ones((A4_H, A4_W, 3), dtype=np.uint8) * 240
    for obj_id in range(1, n_objects + 1):
        color = COLORS[(obj_id-1) % len(COLORS)]
        seg_vis[result_map == obj_id] = color
        mask   = (result_map == obj_id).astype(np.uint8)
        coords = cv2.findNonZero(mask)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            cv2.putText(seg_vis, str(obj_id),
                        (x+w//2-10, y+h//2+10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                        (255,255,255), 3, cv2.LINE_AA)

    def make_panel(img, label):
        bgr  = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim==2 else img.copy()
        sc   = TARGET_H / bgr.shape[0]
        p    = cv2.resize(bgr, (int(bgr.shape[1]*sc), TARGET_H))
        cv2.putText(p, label, (5,24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,0), 2, cv2.LINE_AA)
        return p

    row1 = np.hstack([
        make_panel(orig,      "0.Original"),
        make_panel(paper_vis, "1.Paper"),
        make_panel(warped,    "2.Warped A4"),
        make_panel(binary,    "3.Binary"),
        make_panel(seg_vis,   f"4.Segmented ({n_objects}obj)"),
    ])

    obj_panels = []
    for i, path in enumerate(saved_paths):
        obj_img = cv2.imread(path)
        if obj_img is not None:
            obj_panels.append(make_panel(obj_img, f"obj{i+1}"))

    if obj_panels:
        row2  = np.hstack(obj_panels)
        w     = min(row1.shape[1], row2.shape[1])
        row1  = cv2.resize(row1, (w, TARGET_H))
        row2  = cv2.resize(row2, (w, TARGET_H))
        final = np.vstack([row1, row2])
    else:
        final = row1

    bar = np.zeros((36, final.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar,
                f"Objects: {n_objects}  |  Saved: {len(saved_paths)}  |  A4: {A4_W}x{A4_H}px",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0,255,200), 1, cv2.LINE_AA)
    final = np.vstack([final, bar])

    cv2.imshow("Object Segmentation (Auto)", final)
    print("\n[시각화] 아무 키나 누르면 종료")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ── 메인 ─────────────────────────────────────────────────────────────
def process(
    image_path: str,
    merge_dist: int,
    output_base: str,
    no_paper: bool,
    no_show: bool = False,
) -> tuple[list[str], str]:
    """이미지 → 객체 분할 메인 파이프라인.

    처리 흐름:
    1. 종이 검출 + 원근 변환 (no_paper 시 스킵)
    2. 이진화 (binarize)
    3. 연결 컴포넌트 분석 + Union-Find 병합 (auto_segment)
    4. 객체별 크롭 + JPG 저장 (extract_and_save)

    Args:
        image_path: 입력 이미지 경로
        merge_dist: 컴포넌트 병합 거리 임계값 (픽셀)
        output_base: 결과 저장 기준 폴더 (하위에 segments/ 생성)
        no_paper: True면 종이 검출 스킵
        no_show: True면 시각화 창 표시 안 함

    Returns:
        (저장된 파일 경로 목록, 저장 폴더 경로) 튜플
    """
    print(f"\n이미지 로드: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"[오류] 읽기 실패: {image_path}"); sys.exit(1)

    orig = img.copy()
    print(f"[원본] {img.shape[1]}×{img.shape[0]}px")

    output_dir = os.path.join(output_base, "segments")
    print(f"[저장폴더] {output_dir}")

    if no_paper:
        warped    = cv2.resize(img, (A4_W, A4_H))
        paper_vis = img.copy()
        print("[종이 검출] 스킵")
    else:
        warped, paper_vis = detect_paper(img)

    print("[1] 이진화...")
    binary = binarize(warped)

    print(f"[2] 자동 객체 분할 (merge_dist={merge_dist}px)...")
    result_map, n_objects = auto_segment(binary, merge_dist)
    if result_map is None:
        sys.exit(1)

    base_name   = os.path.splitext(os.path.basename(image_path))[0]
    saved_paths = extract_and_save(
        warped, result_map, n_objects, output_dir, base_name
    )

    print(f"\n[완료] {n_objects}개 객체 → {len(saved_paths)}개 저장")
    for p in saved_paths:
        print(f"  → {p}")

    if not no_show:
        visualize(orig, paper_vis, warped, binary,
                  result_map, n_objects, saved_paths)

    return saved_paths, output_dir


# ── CLI ───────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    """CLI 인자를 파싱하여 반환."""
    p = argparse.ArgumentParser(
        description='이미지 → 자동 객체 분할 v3 (테두리 노이즈 제거)'
    )
    p.add_argument('--image',      required=True)
    p.add_argument('--merge-dist', type=int, default=150,
                   help='컴포넌트 병합 거리 px (기본 150)')
    p.add_argument('--output',     default='')
    p.add_argument('--no-paper',   action='store_true')
    p.add_argument('--no-show',    action='store_true',
                   help='시각화 창 표시 안 함')
    return p.parse_args()


if __name__ == '__main__':
    args       = parse_args()
    output_base = args.output or os.path.join(
        os.path.dirname(os.path.abspath(args.image)), 'objects'
    )
    process(
        image_path  = args.image,
        merge_dist  = args.merge_dist,
        output_base = output_base,
        no_paper    = args.no_paper,
        no_show     = args.no_show,
    )
