# config.py
import os

BASE_DIR: str = os.path.dirname(os.path.dirname(__file__))

SAVE_DIR: str = os.path.join(BASE_DIR, 'captures')
os.makedirs(SAVE_DIR, exist_ok=True)

AUTO_SAVE_SECONDS: float = 5.0   # 사각형 감지 후 자동 캡처까지의 대기 시간 (초)
COOLDOWN_SECONDS: float  = 3.0   # 캡처 직후 재감지 방지를 위한 쿨다운 (초)
MIN_CONTOUR_AREA: int    = 10_000 # 이 면적 미만 컨투어는 노이즈로 간주하여 무시 (픽셀²)
NO_DETECT_TOLERANCE: float = 0.4  # 감지 실패 후 상태 초기화 전 허용 유예 시간 (초)
TARGET_SIZE: tuple[int, int] = (800, 600)  # 원근 변환 후 저장할 이미지 해상도 (픽셀)
CAMERA_INDEX: int = 1             # OpenCV VideoCapture에 전달할 카메라 장치 번호