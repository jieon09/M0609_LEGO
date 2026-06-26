import cv2
import numpy as np
import os
from datetime import datetime
from typing import Optional, Generator, Tuple

from .config import MIN_CONTOUR_AREA, TARGET_SIZE, SAVE_DIR

class ImageProcessor:
    """OpenCV 기반 이미지 전처리, 윤곽 검출, 원근 변환, 저장 기능 제공."""

    # 밝기 대비 향상용 (전역 대비보다 로컬 대비 강화)
    _clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    def edge_map(self, frame: np.ndarray) -> np.ndarray:
        """엣지 맵 생성 (Canny + Adaptive Threshold 혼합).

        처리 흐름:
        1. HSV V 채널 추출 → CLAHE 적용
        2. Gaussian Blur
        3. median 기반 Canny edge
        4. Adaptive threshold
        5. OR 결합 → Morphology Close

        Args:
            frame: BGR 이미지 (HxWx3)

        Returns:
            이진 엣지 맵 (HxW)
        """
        v = self._clahe.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 2])
        blurred = cv2.GaussianBlur(v, (7, 7), 0)

        median = float(np.median(blurred))
        lo, hi = max(0, int(median * 0.25)), min(255, int(median * 0.9))
        canny = cv2.Canny(blurred, lo, hi)

        gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_g = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blur_g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4
        )

        combined = cv2.bitwise_or(canny, thresh)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        return cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)

    def _iter_rects(
        self, edges: np.ndarray
    ) -> Generator[Tuple[np.ndarray, float, np.ndarray, float], None, None]:
        """엣지 맵에서 사각형 후보 컨투어를 generator로 반환.

        Returns:
            (contour, area, approx, epsilon) 튜플
        """
        frame_area = edges.shape[0] * edges.shape[1]
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
            area = cv2.contourArea(c)
            if area < MIN_CONTOUR_AREA or area > frame_area * 0.97:
                continue

            hull = cv2.convexHull(c)
            peri = cv2.arcLength(hull, True)

            for eps in (0.02, 0.04, 0.06, 0.08):
                approx = cv2.approxPolyDP(hull, eps * peri, True)
                if len(approx) == 4:
                    yield c, area, approx, eps
                    break

    def find_rect(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """프레임에서 첫 번째로 발견된 사각형 contour 반환.

        Returns:
            4점 contour 또는 None
        """
        edges = self.edge_map(frame)
        for _, _, approx, _ in self._iter_rects(edges):
            return approx
        return None

    def perspective_transform(self, image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """4점 기반 원근 변환 (document scan 방식).

        처리 흐름:
        1. 점 정렬 (TL, TR, BR, BL)
        2. 출력 크기 계산
        3. 변환 행렬 계산 → warpPerspective 적용

        Args:
            image: 원본 BGR 이미지
            pts: 4개의 코너 좌표 (4x2 float32)

        Returns:
            원근 변환된 이미지
        """
        rect = self._order_points(pts)
        tl, tr, br, bl = rect

        width  = max(int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))), 1)
        height = max(int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))), 1)

        dst = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype='float32'
        )
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, M, (width, height))

    def debug_frame(self, raw: np.ndarray) -> np.ndarray:
        """엣지 + 컨투어 + 통계 정보를 시각화한 디버그 프레임 생성."""
        edges = self.edge_map(raw)
        dbg = raw.copy()

        edge_color = np.zeros_like(raw)
        edge_color[edges > 0] = (180, 60, 0)
        dbg = cv2.addWeighted(dbg, 0.6, edge_color, 0.8, 0)

        for c, area, approx, eps in self._iter_rects(edges):
            cv2.drawContours(dbg, [c], -1, (0, 0, 255), 2)
            cv2.drawContours(dbg, [approx], -1, (0, 255, 0), 3)
            cv2.putText(
                dbg, f'area={int(area)} eps={eps}',
                approx[0][0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
            )

        h = raw.shape[0]
        median_v = int(np.median(cv2.cvtColor(raw, cv2.COLOR_BGR2HSV)[:, :, 2]))
        cv2.putText(
            dbg, f'median_v={median_v}',
            (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
        )
        return dbg

    def save_warped(self, frame: np.ndarray, contour: np.ndarray, prefix: str = 'auto') -> str:
        """원근 변환 후 리사이즈하여 파일 저장.

        Returns:
            저장된 파일명 (SAVE_DIR 기준)
        """
        warped   = self.perspective_transform(frame, contour.reshape(4, 2).astype('float32'))
        result   = cv2.resize(warped, TARGET_SIZE)
        filename = datetime.now().strftime(f'{prefix}_%Y%m%d_%H%M%S.png')
        cv2.imwrite(os.path.join(SAVE_DIR, filename), result)
        return filename

    @staticmethod
    def _order_points(pts: np.ndarray) -> np.ndarray:
        """4개의 점을 [좌상, 우상, 우하, 좌하] 순으로 정렬."""
        rect = np.zeros((4, 2), dtype='float32')
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect
