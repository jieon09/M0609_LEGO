import threading
from typing import Any

# 음성 일시정지 공유 상태 딕셔너리
# voice_detector(ROS 콜백)와 automation_service(메인 루프)가 동시 접근하므로 voice_lock으로 보호
voice_state: dict[str, Any] = {
    "paused": False,  # True = 정지 키워드 감지 → 로봇 일시정지
}

# voice_state 접근 시 반드시 사용해야 하는 전역 락
voice_lock = threading.Lock()
