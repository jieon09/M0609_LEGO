import threading
from typing import Any

# 손 감지 공유 상태 딕셔너리
# hand_subscriber(ROS 콜백)와 automation_service(메인 루프)가 동시 접근하므로 hand_lock으로 보호
hand_state: dict[str, Any] = {
    "detected": False,  # True = 손 감지 중 → 로봇 정지 및 자동화 대기
}

# hand_state 접근 시 반드시 사용해야 하는 전역 락
hand_lock = threading.Lock()
