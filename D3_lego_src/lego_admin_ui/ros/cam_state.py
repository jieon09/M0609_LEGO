import threading
from typing import Any

# 카메라·YOLO 공유 상태 딕셔너리
# 여러 스레드(카메라 노드, Flask 핸들러)가 동시에 접근하므로 반드시 cam_lock으로 보호
cam_state: dict[str, Any] = {
    "yolo_on":    False,       # YOLO 추론 활성화 여부
    "frame":      b"",        # JPEG 인코딩된 최신 컬러 프레임 (bytes)
    "detections": [],          # YOLO 탐지 결과 목록 (dict 리스트)
    "depth_frame": None,       # 깊이 이미지 배열 (np.ndarray, mm 단위)
    "cam_info":   None,        # 카메라 내부 파라미터 {fx, fy, ppx, ppy}
}

# cam_state 접근 시 반드시 사용해야 하는 전역 락
cam_lock = threading.Lock()
