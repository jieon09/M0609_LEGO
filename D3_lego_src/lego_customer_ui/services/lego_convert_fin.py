"""
lego_convert.py  v3
───────────────────
변경사항:
  - 사용 색상: R(빨강) / B(파랑) / Y(노랑) 3색만
  - 선 색상 자동 감지: HSV로 선 색상 분석 → 자동으로 R/B/Y 브릭 매핑
  - Erode / Dilate 독립 적용
  - 그림 영역 자동 크롭 + 24×24 꽉 채우기
  - --no-paper 옵션

사용법:
  python3 lego_convert.py --image ~/Downloads/IOS_heart.jpg
  python3 lego_convert.py --image ~/Downloads/smile.jpg --no-paper
  python3 lego_convert.py --image ~/Downloads/IOS_heart.jpg --outline B
"""

import argparse
import os
import json
import sys
from typing import Optional
import cv2
import numpy as np


# ── 선 색상 자동 감지 ─────────────────────────────────────────────────
def detect_line_color(img: np.ndarray) -> str:
    """
    원본 컬러 이미지에서 선 색상 감지 (이진화 전에 호출)

    처리 흐름:
    1. HSV V채널 기준으로 어두운 픽셀(선 영역) 추출
    2. 해당 픽셀들의 Hue 분포로 R/B/Y 중 가장 많은 색상 선택

    Args:
        img: BGR 이미지 (이진화 전 원본)

    Returns:
        'R', 'B', 'Y' 중 하나. 무채색이거나 매칭 없으면 기본값 'R'
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # V < 100 인 픽셀을 선(어두운 영역)으로 간주 — 100은 경험적 임계값   
    s_channel = hsv[:,:,1]
    color_mask = s_channel > 50# 유채색 픽셀 (S 기준)
    if not np.any(color_mask):
        return 'R'

    h_vals = hsv[:,:,0][color_mask]

    mean_h = np.mean(h_vals)
    print(f"평균 Hue:{mean_h:.1f}")
    

    # HSV Hue 범위 (OpenCV: 0~180)
    # 빨강: 0~10 또는 160~180 (Hue가 원형이라 양 끝에 걸침)
    # 노랑: 20~35
    # 파랑: 100~130
    blue_mask   = ((h_vals >= 100) & (h_vals <= 130))
    yellow_mask = ((h_vals >= 20)  & (h_vals <= 35))
    red_mask    = ((h_vals <= 10)  | (h_vals >= 160))

    counts = {
        'R': np.sum(red_mask),
        'B': np.sum(blue_mask),
        'Y': np.sum(yellow_mask),
    }
    print(f"[색상 감지] R:{counts['R']} B:{counts['B']} Y:{counts['Y']}")

    # 가장 많은 픽셀 수의 색상 선택 — 동점이면 max()가 딕셔너리 삽입 순서(R) 반환
    best = max(counts, key=counts.get)
    if counts[best] == 0:
        print("[색상 감지] 매칭 없음 → 기본값 R")
        return 'R'

    print(f"[색상 감지] 자동 선택 → {best}")
    return best


# ── EdgeDetector ──────────────────────────────────────────────────────
class EdgeDetector:
    """HSV V채널 + CLAHE + Adaptive Threshold 기반 이진화기."""

    def __init__(self, block_size: int = 21, c: int = 15) -> None:
        self.block_size = block_size  # Adaptive Threshold 블록 크기 (홀수)
        self.c = c                    # Adaptive Threshold 상수 — 클수록 더 적은 엣지

    def detect(self, img: np.ndarray) -> np.ndarray:
        """BGR 이미지를 이진 엣지 맵으로 변환.

        처리 흐름:
        1. HSV V채널 추출 → CLAHE로 로컬 대비 강화
        2. Gaussian Blur → Adaptive Threshold
        3. Morphology Close로 끊긴 선 연결

        Args:
            img: BGR 이미지

        Returns:
            이진 맵 (HxW, 0 또는 255)
        """
        hsv     = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        _, _, v = cv2.split(hsv)
        clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        v       = clahe.apply(v)
        blurred = cv2.GaussianBlur(v, (11, 11), 0)
        thresh  = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.block_size, self.c
        )
        kernel = np.ones((3, 3), np.uint8)
        return cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)


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


def four_point_transform(img: np.ndarray, pts: np.ndarray, size: int = 300) -> np.ndarray:
    """4점 기반 원근 변환으로 정사각형 이미지 생성.

    Args:
        img: BGR 원본 이미지
        pts: 4개의 코너 좌표 (이미지 픽셀 좌표)
        size: 출력 이미지 한 변 크기 (픽셀)

    Returns:
        size x size BGR 이미지
    """
    rect = order_points(pts)
    dst  = np.array([[0,0],[size-1,0],[size-1,size-1],[0,size-1]],
                    dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (size, size))


def detect_paper(img: np.ndarray, SZ: int = 300) -> tuple[np.ndarray, np.ndarray]:
    """이미지에서 종이 영역을 검출하고 원근 변환 적용.

    처리 흐름:
    1. Otsu 이진화로 가장 큰 윤곽선 검출
    2. 꼭짓점 4개이면 원근 변환, 아니면 중앙 크롭

    Args:
        img: BGR 원본 이미지
        SZ: 출력 크기 (픽셀). 기본 300

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
        print("[종이 검출] 윤곽선 없음 → 원본 사용")
        return cv2.resize(img, (SZ, SZ)), paper_vis

    largest = max(contours, key=cv2.contourArea)
    peri    = cv2.arcLength(largest, True)
    approx  = cv2.approxPolyDP(largest, 0.02 * peri, True)

    cv2.drawContours(paper_vis, [approx], -1, (0, 255, 0), 2)
    for pt in approx.reshape(-1, 2):
        cv2.circle(paper_vis, tuple(pt.astype(int)), 8, (0, 0, 255), -1)

    if len(approx) == 4:
        pts    = approx.reshape(4, 2).astype(np.float32)
        warped = four_point_transform(img, pts, SZ)
        print("[종이 검출] 성공 → 원근 변환 적용")
        return warped, paper_vis
    else:
        # 4점 미만이면 완전한 사각형이 아님 → 10% 마진 크롭으로 테두리 노이즈 제거
        print(f"[종이 검출] 꼭짓점 {len(approx)}개 → 중앙 크롭")
        m = int(SZ * 0.1)
        resized = cv2.resize(img, (SZ, SZ))
        return resized[m:SZ-m, m:SZ-m], paper_vis


# ── 자동 크롭 ─────────────────────────────────────────────────────────
def auto_crop_to_drawing(
    binary: np.ndarray, img_bgr: np.ndarray, padding: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """이진 맵의 비제로 영역 바운딩 박스로 정사각형 크롭.

    Args:
        binary: 이진 맵 (HxW)
        img_bgr: 대응하는 BGR 이미지
        padding: 바운딩 박스에 추가할 여백 (픽셀)

    Returns:
        (크롭된 이진 맵, 크롭된 BGR 이미지) 튜플
        비제로 픽셀이 없으면 입력 그대로 반환
    """
    coords = cv2.findNonZero(binary)
    if coords is None:
        return binary, img_bgr
    x, y, w, h = cv2.boundingRect(coords)
    H, W = binary.shape
    x1 = max(0, x - padding);    y1 = max(0, y - padding)
    x2 = min(W, x + w + padding); y2 = min(H, y + h + padding)
    cw, ch = x2-x1, y2-y1
    # 레고 그리드가 정사각형이므로 크롭 영역도 정사각형으로 보정
    if cw != ch:
        diff = abs(cw - ch)
        if cw < ch:
            x1 = max(0, x1-diff//2); x2 = min(W, x2+diff//2)
        else:
            y1 = max(0, y1-diff//2); y2 = min(H, y2+diff//2)
    print(f"[자동 크롭] ({x1},{y1})-({x2},{y2})")
    return binary[y1:y2, x1:x2], img_bgr[y1:y2, x1:x2]


# ── 내부 채우기 ───────────────────────────────────────────────────────
def flood_fill_interior(binary: np.ndarray) -> np.ndarray:
    """외곽 선으로 둘러싸인 내부 영역을 채워 내부 픽셀 마스크 생성.

    처리 흐름:
    1. bitwise_not으로 배경과 내부를 흰색으로 전환
    2. 좌상단(0,0)에서 floodFill(128) — 배경만 128로 채워짐
    3. 128이 아닌 흰색(255) 픽셀 = 선 내부 영역

    Args:
        binary: 이진 엣지 맵 (HxW)

    Returns:
        내부 영역만 255인 이진 마스크 (HxW)
    """
    h, w  = binary.shape
    mask  = np.zeros((h+2, w+2), np.uint8)
    inv   = cv2.bitwise_not(binary)
    flood = inv.copy()
    cv2.floodFill(flood, mask, (0, 0), 128)
    out = np.zeros_like(binary)
    out[flood == 255] = 255
    return out


# ── Sobel 엣지 ────────────────────────────────────────────────────────
def sobel_edge(binary: np.ndarray) -> np.ndarray:
    """이진 맵에서 Sobel 미분으로 엣지(경계선) 추출.

    Args:
        binary: 이진 맵 (HxW)

    Returns:
        엣지 이진 맵 — 최대 gradient의 12% 이상인 픽셀만 255
        gradient 최대값이 0이면 빈 맵 반환
    """
    bf  = binary.astype(np.float32)
    gx  = cv2.Sobel(bf, cv2.CV_32F, 1, 0, ksize=3)
    gy  = cv2.Sobel(bf, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    mx  = mag.max()
    if mx == 0: return np.zeros_like(binary)
    out = np.zeros_like(binary)
    out[mag > mx * 0.12] = 255  # 0.12: 약한 노이즈 엣지 제거를 위한 경험적 임계 비율
    return out


# ── 24×24 다운샘플 ────────────────────────────────────────────────────
def downsample_24x24(
    colored: np.ndarray,
    SZ: int,
    outline_color: str,
    fill_color: str,
) -> list[str]:
    """colored 맵을 24×24 레고 그리드로 다운샘플링.

    각 셀에서 엣지(1) / 내부(2) 픽셀 비율을 계산하여 브릭 색상 코드 할당.

    Args:
        colored: 픽셀값 1(엣지) 또는 2(내부) 또는 0(배경) (SZxSZ)
        SZ: colored 이미지 한 변 크기 (픽셀)
        outline_color: 엣지 픽셀에 할당할 색상 코드 ('R'/'B'/'Y')
        fill_color: 내부 픽셀에 할당할 색상 코드 ('R'/'B'/'Y' 또는 '.')

    Returns:
        길이 576(24×24)의 색상 코드 리스트 ('R'/'B'/'Y'/'.')
    """
    CS = SZ / 24  # 셀 하나의 픽셀 크기
    grid: list[str] = []
    for r in range(24):
        for c in range(24):
            r0=int(round(r*CS));      c0=int(round(c*CS))
            r1=min(int(round((r+1)*CS)),SZ)
            c1=min(int(round((c+1)*CS)),SZ)
            cell=colored[r0:r1,c0:c1]; total=cell.size
            if total==0: grid.append('.'); continue
            er=np.sum(cell==1)/total; ir=np.sum(cell==2)/total
            # 0.06: 셀의 6% 이상이 엣지이면 외곽선 브릭으로 판단
            if   er>0.06:                         grid.append(outline_color)
            # 0.20: 셀의 20% 이상이 내부이면 채우기 브릭으로 판단
            elif ir>0.20 and fill_color!='.':     grid.append(fill_color)
            else:                                  grid.append('.')
    return grid


# ── 작은 클러스터 제거 ────────────────────────────────────────────────
def remove_small_clusters(grid: list[str], min_cells: int) -> list[str]:
    """BFS로 연결된 동색 클러스터를 탐색하여 min_cells 미만 클러스터를 '.'으로 대체.

    Args:
        grid: 24×24 색상 코드 리스트
        min_cells: 이 셀 수 미만의 클러스터는 노이즈로 제거

    Returns:
        소규모 클러스터가 제거된 그리드
    """
    N=24; visited=[False]*(N*N); result=grid[:]
    for i in range(N*N):
        if visited[i] or grid[i]=='.': continue
        color=grid[i]; comp=[]; stack=[i]
        while stack:
            idx=stack.pop()
            if visited[idx]: continue
            visited[idx]=True; comp.append(idx)
            r,c=divmod(idx,N)
            for dr,dc in[(-1,0),(1,0),(0,-1),(0,1)]:
                nr,nc=r+dr,c+dc
                if 0<=nr<N and 0<=nc<N:
                    ni=nr*N+nc
                    if not visited[ni] and grid[ni]==color:
                        stack.append(ni)
        if len(comp)<min_cells:
            for idx in comp: result[idx]='.'
    return result


# ── 그리디 브릭 변환 (R/B/Y 만) ──────────────────────────────────────
def make_bricks(grid: list[str]) -> tuple[list[dict], int]:
    """24×24 그리드를 2×2, 2×3, 3×2 브릭으로 그리디 변환.

    처리 흐름:
    1. 좌상→우하 순으로 스캔
    2. 미사용 셀에서 가장 큰 브릭(2×3 → 3×2 → 2×2) 우선 배치
    3. R/B/Y 이외 색상 및 단독 셀은 dropped 처리

    Args:
        grid: 24×24 색상 코드 리스트

    Returns:
        (브릭 목록, 버려진 셀 수) 튜플
    """
    N=24; used=[[False]*N for _ in range(N)]
    bricks: list[dict] = []; dropped=0

    def can(r: int, c: int, w: int, h: int, color: str) -> bool:
        if r+h>N or c+w>N: return False
        for dr in range(h):
            for dc in range(w):
                if used[r+dr][c+dc] or grid[(r+dr)*N+(c+dc)]!=color:
                    return False
        return True

    def place(r: int, c: int, w: int, h: int, color: str) -> None:
        for dr in range(h):
            for dc in range(w): used[r+dr][c+dc]=True
        bricks.append({'row':r,'col':c,'width':w,'height':h,'color':color})

    for r in range(N):
        for c in range(N):
            if used[r][c]: continue
            color=grid[r*N+c]
            if color=='.': continue
            # R/B/Y 만 허용 — 다른 값은 그리드 오류로 무시
            if color not in ('R','B','Y'):
                used[r][c]=True; dropped+=1; continue
            if   can(r,c,2,3,color): place(r,c,2,3,color)
            elif can(r,c,3,2,color): place(r,c,3,2,color)
            elif can(r,c,2,2,color): place(r,c,2,2,color)
            else: used[r][c]=True; dropped+=1
    return bricks, dropped


# ── 시각화 패널 ───────────────────────────────────────────────────────
def make_panel(img: np.ndarray, label: str, target_h: int = 300) -> np.ndarray:
    """단일 이미지를 target_h 높이로 리사이즈하고 레이블 텍스트를 추가."""
    bgr  = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR) if img.ndim==2 else img.copy()
    sc   = target_h/bgr.shape[0]
    panel= cv2.resize(bgr,(int(bgr.shape[1]*sc),target_h))
    cv2.putText(panel,label,(5,24),cv2.FONT_HERSHEY_SIMPLEX,
                0.65,(0,255,0),2,cv2.LINE_AA)
    return panel


def make_grid_vis(grid: list[str], cell: int = 10) -> np.ndarray:
    """24×24 색상 그리드를 컬러 격자 이미지로 시각화."""
    # BGR 색상: R=빨강 B=파랑 Y=노랑
    CMAP={'R':(0,0,220),'B':(200,80,0),'Y':(0,200,240),'.':(40,40,40)}
    vis=np.zeros((24*cell,24*cell,3),dtype=np.uint8)
    for r in range(24):
        for c in range(24):
            col=CMAP.get(grid[r*24+c],(40,40,40))
            cv2.rectangle(vis,(c*cell,r*cell),(c*cell+cell-1,r*cell+cell-1),col,-1)
            cv2.rectangle(vis,(c*cell,r*cell),(c*cell+cell-1,r*cell+cell-1),(80,80,80),1)
    return vis


def make_brick_vis(bricks: list[dict], cell: int = 10) -> np.ndarray:
    """브릭 목록을 격자 이미지로 시각화."""
    BMAP={'R':(0,0,220),'B':(200,80,0),'Y':(0,200,240)}
    vis=np.ones((24*cell,24*cell,3),dtype=np.uint8)*235
    for b in bricks:
        x1,y1=b['col']*cell,b['row']*cell
        x2,y2=x1+b['width']*cell-1,y1+b['height']*cell-1
        bc=BMAP.get(b['color'],(100,100,100))
        cv2.rectangle(vis,(x1,y1),(x2,y2),bc,-1)
        cv2.rectangle(vis,(x1,y1),(x2,y2),(50,50,50),1)
    return vis


# ── 메인 파이프라인 ───────────────────────────────────────────────────
def process(
    image_path: str,
    output_path: str,
    outline_color: str,
    fill_color: str,
    dilate_amt: int,
    simplify: int,
    min_cluster: int,
    no_paper: bool,
    auto_color: bool,
    no_show: bool = False,
) -> dict:

    """이미지 → 레고 브릭 JSON 변환 메인 파이프라인.

    처리 흐름:
    0. 종이 검출 + 원근 변환 (no_paper 옵션 시 스킵)
    1. EdgeDetector 이진화
    2. Erode (선 세기 조절)
    3. Dilate (선 두께 복원)
    4. Sobel 엣지 추출
    5. Flood Fill 내부 채우기
    6. 24×24 다운샘플
    7. 소규모 클러스터 제거
    8. 그리디 브릭 변환 + JSON 저장

    Args:
        image_path: 입력 이미지 경로
        output_path: 출력 JSON 파일 경로
        outline_color: 엣지 브릭 색상 ('R'/'B'/'Y')
        fill_color: 내부 브릭 색상 ('R'/'B'/'Y' 또는 '.')
        dilate_amt: Dilate 반복 횟수
        simplify: Erode 반복 횟수 (선 세기 줄이기)
        min_cluster: 이 셀 수 미만 클러스터 제거
        no_paper: True면 종이 검출 스킵
        auto_color: True면 HSV 분석으로 색상 자동 감지
        no_show: True면 시각화 창 표시 안 함

    Returns:
        변환 결과 딕셔너리 (brick_count, bricks 등)
    """
    SZ = 300  # 처리 해상도: 300×300px (고정) — 24×24 그리드의 정수 배율 기준
    print(f"\n이미지 로드: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"[오류] 이미지 읽기 실패: {image_path}"); sys.exit(1)
    img  = cv2.resize(img, (SZ, SZ))
    orig = img.copy()

    # Step 0: 종이 검출
    if no_paper:
        warped    = img.copy()
        paper_vis = img.copy()
        cv2.putText(paper_vis,"no-paper mode",(5,24),
                    cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,200,255),2)
        print("[종이 검출] 스킵")
    else:
        warped, paper_vis = detect_paper(img, SZ)

    # 색상 자동 감지는 이진화 전 원본 컬러 이미지에서 수행해야 정확
    if auto_color:
        outline_color = detect_line_color(warped)
        print(f"[색상 자동감지] outline → {outline_color}")
    else:
        print(f"[색상 수동지정] outline → {outline_color}")

    # Step 1: EdgeDetector (이진화)
    detector    = EdgeDetector(block_size=21, c=15)
    binary      = detector.detect(warped)
    binary_orig = binary.copy()

    # ── 자동 크롭 ─────────────────────────────────────────────────────
    binary_crop, warped_crop = auto_crop_to_drawing(binary, warped, padding=10)
    binary = cv2.resize(binary_crop, (SZ,SZ), interpolation=cv2.INTER_NEAREST)
    warped = cv2.resize(warped_crop, (SZ,SZ))

    # Step 2: Erode (독립) — 잔선·노이즈 제거용, simplify=0이면 건너뜀
    proc = binary.copy()
    if simplify > 0:
        k = np.ones((3,3),np.uint8)
        proc = cv2.erode(proc, k, iterations=simplify)
    erode_vis = proc.copy()

    # Step 3: Dilate (독립) — erode로 얇아진 선을 다시 굵게, dilate_amt=0이면 건너뜀
    if dilate_amt > 0:
        k = np.ones((3,3),np.uint8)
        proc = cv2.dilate(proc, k, iterations=dilate_amt)

    # Step 4: Sobel
    edges = sobel_edge(proc)

    # Step 5: Flood Fill — 엣지 = 1, 내부 = 2, 배경 = 0
    interior = flood_fill_interior(proc)
    colored  = np.zeros((SZ,SZ),dtype=np.uint8)
    colored[interior>0] = 2
    colored[edges>0]    = 1

    colored_vis = np.zeros((SZ,SZ,3),dtype=np.uint8)
    colored_vis[colored==1] = (0,0,255)
    colored_vis[colored==2] = (0,200,200)

    # Step 6: 24×24 다운샘플
    grid = downsample_24x24(colored, SZ, outline_color, fill_color)

    # Step 7: 클러스터 제거
    grid     = remove_small_clusters(grid, min_cluster)
    grid_vis = make_grid_vis(grid)

    # Step 8: 브릭 변환 (R/B/Y 만)
    bricks, dropped = make_bricks(grid)
    brick_vis       = make_brick_vis(bricks)

    # 통계
    cnt={'R':0,'B':0,'Y':0}; t22=t23=t32=0; cells=0
    for b in bricks:
        if b['color'] in cnt: cnt[b['color']]+=1
        cells+=b['width']*b['height']
        if   b['width']==2 and b['height']==2: t22+=1
        elif b['width']==2 and b['height']==3: t23+=1
        elif b['width']==3 and b['height']==2: t32+=1
    cov = round(cells/576*100)

    design = {
        'source': image_path, 'brick_count': len(bricks),
        'coverage_pct': cov,
        'brick_types': {'2x2':t22,'2x3':t23,'3x2':t32},
        'dropped_cells': dropped, 'colors': cnt,
        'detected_line_color': outline_color,
        'bricks': bricks
    }

    print(f"\n[결과] 브릭:{len(bricks)} | 커버리지:{cov}% | 버림:{dropped}")
    print(f"       R:{cnt['R']} B:{cnt['B']} Y:{cnt['Y']}")
    print(f"       2×2:{t22} 2×3:{t23} 3×2:{t32}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path,'w',encoding='utf-8') as f:
        json.dump(design, f, indent=2, ensure_ascii=False)
    print(f"[저장] {output_path}")

    # ── 시각화 창 하나 ────────────────────────────────────────────────
    TARGET_H = 300
    steps = [
        ("0.Original",    orig),
        ("0.Paper",       paper_vis),
        ("0.Warped+Crop", warped),
        ("1.Binary",      binary_orig),
        ("2.Erode",       erode_vis),
        ("3.Dilate+Sobel",edges),
        ("5.Colored",     colored_vis),
        ("6.Grid 24x24",  grid_vis),
        ("8.Bricks",      brick_vis),
    ]

    panels = [make_panel(i, l, TARGET_H) for l, i in steps]
    rows   = []
    for i in range(0, len(panels), 3):
        rp = panels[i:i+3]
        while len(rp)<3:
            rp.append(np.zeros((TARGET_H,TARGET_H,3),dtype=np.uint8))
        rows.append(np.hstack(rp))

    min_w = min(r.shape[1] for r in rows)
    rows  = [cv2.resize(r,(min_w,TARGET_H)) for r in rows]
    disp  = np.vstack(rows)

    # 하단 정보바
    bar = np.zeros((44, disp.shape[1], 3), dtype=np.uint8)
    line1 = (f"Bricks:{len(bricks)}  Coverage:{cov}%  "
             f"R:{cnt['R']} B:{cnt['B']} Y:{cnt['Y']}  "
             f"2x2:{t22} 2x3:{t23} 3x2:{t32}  Dropped:{dropped}")
    line2 = f"Detected line color: {outline_color}  |  outline={outline_color}  fill={fill_color}"
    cv2.putText(bar,line1,(8,14),cv2.FONT_HERSHEY_SIMPLEX,0.50,(0,255,200),1,cv2.LINE_AA)
    cv2.putText(bar,line2,(8,32),cv2.FONT_HERSHEY_SIMPLEX,0.50,(200,200,0),1,cv2.LINE_AA)

    if not no_show:
        cv2.imshow("Lego Pipeline", np.vstack([disp, bar]))
        print("\n[시각화] 아무 키나 누르면 종료")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return design




# ── 배치 처리 ─────────────────────────────────────────────────────────
def batch_process(
    folder: str,
    dilate_amt: int,
    simplify: int,
    min_cluster: int,
    latest: bool,
    no_show: bool = False,
) -> None:
    """폴더 내 obj*.jpg 파일 전체를 일괄 변환하여 JSON 저장.

    Args:
        folder: obj*.jpg가 있는 폴더 경로
        dilate_amt: Dilate 반복 횟수
        simplify: Erode 반복 횟수
        min_cluster: 소규모 클러스터 제거 임계 셀 수
        latest: True면 폴더 내 최신 타임스탬프 서브폴더 자동 선택
        no_show: True면 시각화 창 표시 안 함
    """
    import glob
    # --latest: 최신 타임스탬프 서브폴더 자동 선택
    if latest:
        subs = sorted([d for d in os.listdir(folder)
                       if os.path.isdir(os.path.join(folder, d))])
        if not subs:
            print("[오류] 서브폴더가 없습니다."); sys.exit(1)
        folder = os.path.join(folder, subs[-1])
        print(f"[최신 폴더] {folder}")

    jpg_files = sorted(glob.glob(os.path.join(folder, 'obj*.jpg')))
    if not jpg_files:
        print("[오류] obj*.jpg 파일이 없습니다."); sys.exit(1)

    print("\n[배치] {}개 객체 변환 시작".format(len(jpg_files)))
    json_paths = []

    for img_path in jpg_files:
        base      = os.path.splitext(os.path.basename(img_path))[0]
        json_path = os.path.join(folder, "{}_bricks.json".format(base))
        print("\n변환: {}".format(os.path.basename(img_path)))
        design = process(
            image_path    = img_path,
            output_path   = json_path,
            outline_color = 'R',
            fill_color    = '.',
            dilate_amt    = dilate_amt,
            simplify      = simplify,
            min_cluster   = min_cluster,
            no_paper      = True,
            auto_color    = True,
            no_show       = no_show,
        )
        json_paths.append(json_path)

    summary_path = os.path.join(folder, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'folder': folder,
            'total': len(json_paths),
            'files': [os.path.basename(p) for p in json_paths]
        }, f, indent=2, ensure_ascii=False)

    print("\n[완료] {}개 JSON 생성".format(len(json_paths)))
    for p in json_paths:
        print("  -> {}".format(p))
    print("  -> {}".format(summary_path))

# ── CLI ───────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    """CLI 인자를 파싱하여 반환."""
    p = argparse.ArgumentParser(description='이미지 → 레고 JSON 변환기 v3 (R/B/Y)')
    p.add_argument('--image',    default='', help='단일 이미지 경로')
    p.add_argument('--output',   default='')
    p.add_argument('--batch',    default='', help='objects 폴더 (배치 모드)')
    p.add_argument('--latest',   action='store_true', help='최신 타임스탬프 폴더 자동 선택')
    p.add_argument('--outline',  default='auto', choices=['auto','R','B','Y'])
    p.add_argument('--fill',     default='.', choices=['R','B','Y','.'])
    p.add_argument('--dilate',   type=int, default=3)
    p.add_argument('--simplify', type=int, default=0)
    p.add_argument('--cluster',  type=int, default=1)
    p.add_argument('--no-paper', action='store_true')
    p.add_argument('--no-show', action='store_true', help='시각화 창 표시 안 함')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.batch:
        batch_process(
            folder      = args.batch,
            dilate_amt  = args.dilate,
            simplify    = args.simplify,
            min_cluster = args.cluster,
            latest      = args.latest,
            no_show     = args.no_show,
        )
    elif args.image:
        if not args.output:
            args.output = os.path.join(
                os.path.dirname(os.path.abspath(args.image)), 'bricks.json'
            )
        auto_color    = (args.outline == 'auto')
        outline_color = 'R' if auto_color else args.outline
        process(
            image_path    = args.image,
            output_path   = args.output,
            outline_color = outline_color,
            fill_color    = args.fill,
            dilate_amt    = args.dilate,
            simplify      = args.simplify,
            min_cluster   = args.cluster,
            no_paper      = args.no_paper,
            auto_color    = auto_color,
            no_show       = args.no_show,
        )
    else:
        print("--image 또는 --batch 를 지정하세요.")
        sys.exit(1)
