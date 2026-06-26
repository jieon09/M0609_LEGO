import cv2
import numpy as np
import threading
import time
from typing import Optional, Tuple

from .config import (
    CAMERA_INDEX,
    AUTO_SAVE_SECONDS,
    COOLDOWN_SECONDS,
    NO_DETECT_TOLERANCE,
)
from .shared_state import SharedState
from .image_processor import ImageProcessor


class CameraManager:
    """카메라 캡처, detection 루프, 프레임 publish를 담당."""

    def __init__(self, state: SharedState, proc: ImageProcessor) -> None:
        self._state = state
        self._proc = proc
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._explicit_stop = False

    def ensure_running(self) -> None:
        """카메라 worker thread가 실행 중이 아니면 시작.

        explicit stop 상태에서는 무시 — 주문 등록 후 video_feed 스트림이
        뒤늦게 호출되더라도 카메라가 다시 켜지지 않도록 한다.
        """
        if self._explicit_stop:
            return
        self._stop_event.clear()
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                # daemon=True: 메인 프로세스 종료 시 자동 정리되도록 설정
                self._thread = threading.Thread(target=self._worker, daemon=True)
                self._thread.start()

    def stop(self, explicit: bool = False) -> None:
        """stop_event를 설정해 worker 루프가 다음 이터레이션에서 종료되도록 신호 전달.

        Args:
            explicit: True면 ensure_running 자동 재시작 차단. reset_explicit_stop()
                필요. 주문 등록 같이 '확실히 꺼야 하는' 경우에 사용.
        """
        if explicit:
            self._explicit_stop = True
        self._stop_event.set()

    def reset_explicit_stop(self) -> None:
        """explicit stop 플래그 해제 — 다음 ensure_running 호출 시 재시작 가능."""
        self._explicit_stop = False

    def capture_manual(self) -> Tuple[str, str]:
        """현재 프레임 기반 수동 캡처.

        Returns:
            ('__ready__', 상태 메시지) 또는 ('', 에러 메시지)
        """
        raw = self._state.get_raw()
        if raw is None:
            return '', '카메라 프레임 읽기 실패'

        contour = self._proc.find_rect(raw)
        label = (
            '그림 감지 성공 — 고객 ID 입력 대기 중'
            if contour is not None
            else '사각형 미감지 — 전체 프레임 고객 ID 입력 대기 중'
        )
        self._state.set_pending_capture(raw, contour)
        # 주문 등록 알림창 대기 중 카메라 정지 → dismiss_draw 시 재시작
        self._stop_event.set()
        return '__ready__', label

    def _worker(self) -> None:
        """카메라 프레임을 읽어 detection 루프를 실행하는 worker.

        처리 흐름:
        1. VideoCapture 열기 시도 (실패 시 0.5초 대기 후 재시도)
        2. 프레임 읽기 → SharedState에 raw 프레임 publish
        3. 쿨다운 중이면 OSD만 표시, 아니면 detection 로직 실행
        4. stop_event가 설정되면 루프 종료 및 카메라 해제

        Note:
            daemon 스레드로 실행되므로 메인 프로세스 종료 시 자동 정리됨
        """
        cap: Optional[cv2.VideoCapture] = None
        detect_start: Optional[float] = None    # 사각형 최초 감지 시각 (unix timestamp)
        no_detect_since: Optional[float] = None  # 감지 소실 시각 (유예 시간 계산용)
        cooldown_until: float = 0.0  # 이 unix timestamp 이전에는 쿨다운 상태
        flash_until: float = 0.0     # 캡처 완료 OSD 표시 종료 시각

        while not self._stop_event.is_set():
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(CAMERA_INDEX)
                if not cap.isOpened():
                    time.sleep(0.5)
                    continue

            ok, frame = cap.read()
            if not ok:
                time.sleep(0.03)
                continue

            now = time.time()
            raw = frame.copy()
            self._state.set_raw(raw)

            if now < cooldown_until:
                self._state.update_detection(
                    is_detecting=False, elapsed=0.0,
                    remaining=AUTO_SAVE_SECONDS, cooldown=True
                )
                if now < flash_until:
                    cv2.putText(
                        frame, '캡처됨 — 고객 ID 입력 중...',
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2
                    )
            else:
                self._state.update_detection(cooldown=False)
                (
                    detect_start, no_detect_since, cooldown_until, flash_until
                ) = self._process_detection(
                    frame, raw, now, detect_start, no_detect_since, cooldown_until, flash_until
                )

            self._publish(frame, raw)

        if cap is not None:
            cap.release()

    def _process_detection(
        self,
        frame: np.ndarray,
        raw: np.ndarray,
        now: float,
        detect_start: Optional[float],
        no_detect_since: Optional[float],
        cooldown_until: float,
        flash_until: float,
    ) -> Tuple[Optional[float], Optional[float], float, float]:
        """사각형 detection 상태 업데이트 및 자동 캡처 조건 판단.

        처리 흐름:
        1. contour 검출
        2. 검출 성공 시 경과 시간 계산 → AUTO_SAVE_SECONDS 초과 시 자동 캡처
        3. 검출 실패 시 NO_DETECT_TOLERANCE 초 유예 후 상태 초기화

        Returns:
            (detect_start, no_detect_since, cooldown_until, flash_until)
        """
        h, w = frame.shape[:2]
        contour = self._proc.find_rect(frame)

        if contour is not None:
            no_detect_since = None
            if detect_start is None:
                detect_start = now

            elapsed   = now - detect_start
            remaining = AUTO_SAVE_SECONDS - elapsed
            ratio     = min(elapsed / AUTO_SAVE_SECONDS, 1.0)

            self._state.update_detection(
                is_detecting=True,
                elapsed=round(elapsed, 1),
                remaining=round(max(remaining, 0.0), 1),
            )

            if remaining <= 0.0:
                self._state.set_pending_capture(raw, contour)
                self._state.update_detection(
                    is_detecting=False, elapsed=0.0, remaining=AUTO_SAVE_SECONDS
                )
                detect_start    = None
                no_detect_since = None
                # 주문 등록 알림창 대기 중 카메라 정지 → dismiss_draw 시 재시작
                self._stop_event.set()
            else:
                # ratio 가 클수록 초록→빨강으로 변하여 긴박감 표현
                color = (0, 255, int(255 * (1.0 - ratio)))
                cv2.drawContours(frame, [contour], -1, color, 3)

                # 하단 진행바: bx/by는 이미지 픽셀 좌표 기준
                bx, by, bw, bh = 10, h - 20, w - 20, 10
                cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (40, 40, 40), -1)
                cv2.rectangle(frame, (bx, by), (bx + int(bw * ratio), by + bh), color, -1)
                cv2.putText(
                    frame, f'Detecting... {remaining:.1f}s',
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
                )
        else:
            if detect_start is not None:
                if no_detect_since is None:
                    no_detect_since = now
                elif now - no_detect_since > NO_DETECT_TOLERANCE:
                    detect_start    = None
                    no_detect_since = None

            if detect_start is None:
                self._state.update_detection(
                    is_detecting=False, elapsed=0.0, remaining=AUTO_SAVE_SECONDS
                )

        return detect_start, no_detect_since, cooldown_until, flash_until

    def _publish(self, frame: np.ndarray, raw: np.ndarray) -> None:
        """OSD가 합성된 frame과 디버그 시각화 프레임을 JPEG로 인코딩하여 SharedState에 저장."""
        ok, buf = cv2.imencode('.jpg', frame)
        if ok:
            self._state.set_frame(buf.tobytes())

        ok2, buf2 = cv2.imencode('.jpg', self._proc.debug_frame(raw))
        if ok2:
            self._state.set_debug(buf2.tobytes())
