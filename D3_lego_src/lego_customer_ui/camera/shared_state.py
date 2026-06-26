import threading
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List

import numpy as np

from .config import AUTO_SAVE_SECONDS


class SharedState:
    """멀티스레드 환경에서 프레임, 탐지 상태, 캡처 데이터를 안전하게 공유하는 클래스."""

    def __init__(self) -> None:
        # 각 데이터 도메인별 독립 Lock — 서로 다른 데이터는 병렬 접근 허용
        self._frame_lock   = threading.Lock()
        self._debug_lock   = threading.Lock()
        self._raw_lock     = threading.Lock()
        self._det_lock     = threading.Lock()
        self._draw_lock    = threading.Lock()
        self._capture_lock = threading.Lock()
        self._obj_sel_lock = threading.Lock()

        self._latest_frame: Optional[bytes] = None  # MJPEG 스트리밍용 최신 JPEG 바이트
        self._latest_debug: Optional[bytes] = None  # 디버그 시각화 JPEG 바이트
        self._raw_frame: Optional[np.ndarray] = None  # BGR 원본 프레임 (이미지 픽셀 좌표)

        self._detection: Dict[str, Any] = {
            'is_detecting': False,
            'elapsed':      0.0,
            'remaining':    AUTO_SAVE_SECONDS,
            'cooldown':     False,
        }

        self._pending_draw_file: Optional[str] = None
        self._pending_capture: Optional[tuple] = None
        self._draw_jobs: List[Dict[str, str]] = []

        # 객체 선택 대기 상태: 파이프라인이 웹 UI의 선택을 blocking 대기하기 위해 사용
        self._pending_obj_sel: Optional[List[str]] = None
        self._obj_sel_event: Optional[threading.Event] = None
        self._obj_sel_result: Optional[List[int]] = None

    def set_frame(self, data: bytes) -> None:
        """MJPEG 스트리밍용 최신 프레임을 갱신."""
        with self._frame_lock:
            self._latest_frame = data

    def get_frame(self) -> Optional[bytes]:
        """MJPEG 스트리밍용 최신 JPEG 바이트 반환."""
        with self._frame_lock:
            return self._latest_frame

    def set_debug(self, data: bytes) -> None:
        """디버그 시각화 프레임을 갱신."""
        with self._debug_lock:
            self._latest_debug = data

    def get_debug(self) -> Optional[bytes]:
        """디버그 시각화 JPEG 바이트 반환."""
        with self._debug_lock:
            return self._latest_debug

    def set_raw(self, frame: np.ndarray) -> None:
        """원본 BGR 프레임을 갱신 (이미지 픽셀 좌표계)."""
        with self._raw_lock:
            self._raw_frame = frame

    def get_raw(self) -> Optional[np.ndarray]:
        """원본 BGR 프레임의 복사본 반환 — 외부에서 직접 수정해도 내부 상태 불변."""
        with self._raw_lock:
            return self._raw_frame.copy() if self._raw_frame is not None else None

    def update_detection(self, **kwargs: Any) -> None:
        """탐지 상태 딕셔너리를 부분 업데이트."""
        with self._det_lock:
            self._detection.update(kwargs)

    def get_detection(self) -> Dict[str, Any]:
        """현재 탐지 상태 딕셔너리의 복사본 반환."""
        with self._det_lock:
            return dict(self._detection)

    def set_pending_draw(self, filename: str) -> None:
        """주문 등록 알림창 표시를 위한 파일명 설정."""
        with self._draw_lock:
            self._pending_draw_file = filename

    def get_pending_draw(self) -> Optional[str]:
        """주문 등록 대기 중인 파일명 반환 (없으면 None)."""
        with self._draw_lock:
            return self._pending_draw_file

    def clear_pending_draw(self) -> None:
        """주문 등록 대기 상태를 해제."""
        with self._draw_lock:
            self._pending_draw_file = None

    def set_pending_capture(self, frame: np.ndarray, contour: Optional[np.ndarray]) -> None:
        """캡처 이미지와 윤곽 정보를 저장하고 draw 대기 상태로 전환.

        Args:
            frame: BGR 원본 프레임 (이미지 픽셀 좌표)
            contour: 감지된 사각형 4점 윤곽 (없으면 None)
        """
        with self._capture_lock:
            self._pending_capture = (frame.copy(), contour)
        with self._draw_lock:
            # '__ready__' 는 프론트엔드가 주문 등록 모달을 띄우는 트리거 값
            self._pending_draw_file = '__ready__'

    def get_pending_capture(self) -> Optional[tuple]:
        """대기 중인 캡처 데이터 반환 — (raw BGR 복사본, contour) 튜플."""
        with self._capture_lock:
            if self._pending_capture is None:
                return None
            raw, contour = self._pending_capture
            return (raw.copy(), contour)

    def clear_pending_capture(self) -> None:
        """캡처 대기 상태와 draw 대기 상태를 동시에 해제."""
        with self._capture_lock:
            self._pending_capture = None
        with self._draw_lock:
            self._pending_draw_file = None

    def add_draw_job(self, customer_id: str, filename: str) -> str:
        """그리기 작업을 내부 목록에 추가하고 작업 ID 반환.

        Returns:
            8자리 대문자 UUID 기반 작업 ID
        """
        task_id = str(uuid.uuid4())[:8].upper()
        with self._draw_lock:
            self._draw_jobs.append({
                'id':          task_id,
                'customer_id': customer_id,
                'filename':    filename,
                'created_at':  datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'status':      'pending',
            })
        return task_id

    def set_pending_obj_selection(self, files: List[str]) -> None:
        """객체 선택 대기 상태를 설정하고 Event를 초기화.

        파이프라인 스레드가 이 호출 이후 wait_obj_selection()으로 blocking 대기한다.
        """
        with self._obj_sel_lock:
            self._pending_obj_sel = list(files)
            self._obj_sel_event  = threading.Event()
            self._obj_sel_result = None

    def get_pending_obj_selection(self) -> Optional[List[str]]:
        """선택 대기 중인 객체 이미지 경로 목록 반환 (없으면 None)."""
        with self._obj_sel_lock:
            return self._pending_obj_sel

    def confirm_obj_selection(self, selected_indices: List[int]) -> None:
        """웹 UI에서 선택 확정 시 호출 — 결과를 저장하고 대기 스레드를 깨운다."""
        with self._obj_sel_lock:
            self._obj_sel_result = selected_indices
            if self._obj_sel_event:
                self._obj_sel_event.set()

    def wait_obj_selection(self, timeout: float = 300) -> Optional[List[int]]:
        """객체 선택이 확정될 때까지 최대 timeout 초 동안 blocking 대기.

        Args:
            timeout: 최대 대기 시간 (초). 기본 300초로 설정 — 고객이 선택을 미룰 수 있음

        Returns:
            선택된 인덱스 목록, timeout 초과 시 None
        """
        with self._obj_sel_lock:
            event = self._obj_sel_event
        if event:
            event.wait(timeout=timeout)
        with self._obj_sel_lock:
            return self._obj_sel_result

    def clear_pending_obj_selection(self) -> None:
        """객체 선택 대기 상태를 전부 초기화."""
        with self._obj_sel_lock:
            self._pending_obj_sel = None
            self._obj_sel_event  = None
            self._obj_sel_result = None
