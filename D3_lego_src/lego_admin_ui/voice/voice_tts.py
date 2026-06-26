import logging
import os
from gtts import gTTS

# gtts 내부 DEBUG 로그 억제
logging.getLogger("gtts").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

log = logging.getLogger(__name__)

# 음성 파일을 모듈 디렉터리에 저장해 경로를 실행 위치에 무관하게 고정
_VOICE_PATH: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'voice.mp3')


def speak(text: str) -> None:
    """
    주어진 텍스트를 한국어 TTS로 변환해 즉시 재생한다.

    처리 흐름:
    1. gTTS로 텍스트 → MP3 변환 및 저장
    2. mpg123으로 재생 (blocking — 재생 완료 전까지 반환 안 됨)

    Args:
        text: 음성으로 출력할 한국어 문자열

    Note:
        os.system()은 blocking 호출이므로 재생이 끝날 때까지 호출 스레드가 점유됨.
        자동화 루프에서 호출 시 해당 스텝이 음성 길이만큼 지연될 수 있음.
    """
    log.info("[VOICE] %s", text)
    tts = gTTS(text=text, lang='ko')
    tts.save(_VOICE_PATH)
    # mpg123: -q 옵션으로 배너/진행 출력 억제
    os.system(f"mpg123 -q {_VOICE_PATH}")


if __name__ == "__main__":
    print("타자 입력 (exit 종료)")
    while True:
        text = input(">>> ")
        if text == "exit":
            break
        speak(text)
