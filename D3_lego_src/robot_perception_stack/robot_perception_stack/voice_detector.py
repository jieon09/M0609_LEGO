"""
voice_detector.py
─────────────────────────────────────────────────────────────────────
ROS2 노드: 마이크 → Whisper STT → 키워드 감지 → /pause_state 퍼블리시

퍼블리시 토픽: /pause_state (std_msgs/Bool)
    True  → 정지 키워드 감지 (로봇 일시 정지)
    False → 재개 키워드 감지 (로봇 재개)

실행:
    ros2 run robot_perception_stack voice_detector
"""

import io
import os
import wave
import threading
import numpy as np
import tempfile

import pyaudio
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from openai import OpenAI
from dotenv import load_dotenv
from ament_index_python.packages import get_package_share_directory

# resource/.env 를 패키지 share 경로에서 로드
_ENV_PATH = os.path.join(
    get_package_share_directory("robot_perception_stack"), "resource", ".env"
)
load_dotenv(dotenv_path=_ENV_PATH)

# ─── 마이크 설정 ─────────────────────────────────────────────────────────────
CHUNK       = 4096
CHANNELS    = 1
FMT         = pyaudio.paInt16
RECORD_SECS = 3

# 무음 판별 임계값: RMS가 이 값 미만이면 Whisper 호출 생략 (환각 방지)
# int16 범위(0~32767) 기준, 500 = 약 1.5% 수준의 음량
SILENCE_THRESHOLD = 250

# ─── 키워드 ──────────────────────────────────────────────────────────────────
PAUSE_KEYWORDS  = ["잠깐 멈춰", "잠깐 멈춰줘", "멈춰줘", "멈춰", "멈추어", "멈추", "정지", "스톱", "stop"]
RESUME_KEYWORDS = ["다시 실행해줘", "다시 실행해", "다시 실행", "계속해줘", "계속해", "재개해", "재개", "시작해"]


class VoiceDetector(Node):

    def __init__(self):
        super().__init__("voice_detector")

        # ─────────────────────────────────────────────
        # 음성 인식 결과 publish
        # topic: /pause_state
        # type: Bool (True = 정지, False = 재개)
        # ─────────────────────────────────────────────
        self.pub = self.create_publisher(Bool, "/pause_state", 10)

        # 현재 정지 상태 (중복 publish 방지)
        self._paused = False

        # ─────────────────────────────────────────────
        # OpenAI Whisper 클라이언트 초기화
        # ─────────────────────────────────────────────
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            self.get_logger().error("OPENAI_API_KEY 가 설정되지 않았습니다.")
            raise RuntimeError("OPENAI_API_KEY missing")
        self._client = OpenAI(api_key=api_key)

        # ─────────────────────────────────────────────
        # 마이크 수신 → STT → 키워드 판단을 백그라운드 스레드에서 실행
        # (blocking I/O이므로 ROS spin 스레드와 분리)
        # ─────────────────────────────────────────────
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

        self.get_logger().info("Voice Detector Node Started")
        self.get_logger().info(f"  정지 키워드 : {PAUSE_KEYWORDS}")
        self.get_logger().info(f"  재개 키워드 : {RESUME_KEYWORDS}")

    # ─────────────────────────────────────────────
    # /pause_state publish 헬퍼
    # paused=True  → 정지 (hand_detected=True 와 동일 의미)
    # paused=False → 재개
    # ─────────────────────────────────────────────
    def _publish(self, paused: bool):
        msg = Bool()
        msg.data = paused
        self.pub.publish(msg)
        label = "⏸  일시 정지" if paused else "▶  재개"
        self.get_logger().info(f"[/pause_state] {label} → {paused}")

    # ─────────────────────────────────────────────
    # 마이크 수신 루프 (백그라운드 스레드)
    # 3초 단위로 녹음 → Whisper STT → 키워드 판단 반복
    # ─────────────────────────────────────────────
    def _listen_loop(self):
        audio = pyaudio.PyAudio()

        # 사용 가능한 입력 장치 탐색
        # hw:0,0 (HDA Analog)은 스피커 루프백이 섞이므로 hw:0,6 이상의 DMIC를 우선 선택
        all_inputs = []
        for i in range(audio.get_device_count()):
            try:
                info = audio.get_device_info_by_index(i)
                if info.get("maxInputChannels", 0) > 0:
                    all_inputs.append((i, info))
            except Exception:
                continue

        if not all_inputs:
            self.get_logger().error("사용 가능한 마이크 입력 장치가 없습니다.")
            audio.terminate()
            return

        # hw:0,6 이상 (DMIC) 우선, 없으면 첫 번째 장치 사용
        preferred = [(i, info) for i, info in all_inputs if "hw:0,6" in info["name"] or "hw:0,7" in info["name"]]
        chosen_idx, chosen_info = preferred[0] if preferred else all_inputs[0]
        input_device_index = chosen_idx
        rate = int(chosen_info["defaultSampleRate"])
        self.get_logger().info(
            f"입력 장치 선택: [{chosen_idx}] {chosen_info['name']} "
            f"(채널={chosen_info['maxInputChannels']}, 샘플레이트={rate}Hz)"
        )
        self.get_logger().info(f"전체 입력 장치: {[(i, info['name']) for i, info in all_inputs]}")

        try:
            stream = audio.open(
                format=FMT,
                channels=CHANNELS,
                rate=rate,
                input=True,
                input_device_index=input_device_index,
                frames_per_buffer=CHUNK,
            )
        except OSError as e:
            self.get_logger().error(f"마이크 스트림 열기 실패: {e}")
            audio.terminate()
            return

        sample_width = audio.get_sample_size(FMT)
        num_chunks   = int(rate / CHUNK * RECORD_SECS)

        while not self._stop_event.is_set():
            # 3초 분량 녹음
            frames = []
            for _ in range(num_chunks):
                if self._stop_event.is_set():
                    break
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)

            if not frames:
                continue

            # 무음 체크: RMS 에너지가 임계값 미만이면 Whisper 호출 생략 (환각 방지)
            audio_data = np.frombuffer(b"".join(frames), dtype=np.int16)
            rms = float(np.sqrt(np.mean(audio_data.astype(np.float32) ** 2)))
            if rms < SILENCE_THRESHOLD:
                continue

            # WAV 버퍼 생성
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(sample_width)
                wf.setframerate(rate)
                wf.writeframes(b"".join(frames))

            # Whisper STT
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(wav_buffer.getvalue())
                    tmp_path = tmp.name

                with open(tmp_path, "rb") as f:
                    transcript = self._client.audio.transcriptions.create(
                        model="whisper-1",
                        file=f,
                        language="ko",
                    )
                text = transcript.text.strip()
            except Exception as e:
                self.get_logger().warn(f"Whisper 오류: {e}")
                continue
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)

            if not text:
                continue

            self.get_logger().info(f"인식: \"{text}\"")
            self._check_keywords(text)

        stream.stop_stream()
        stream.close()
        audio.terminate()
        self.get_logger().info("음성 감지 종료")

    # ─────────────────────────────────────────────
    # 키워드 판단
    # 정지 키워드 → True publish (hand_detected=True 와 동일)
    # 재개 키워드 → False publish
    # 이미 같은 상태면 중복 publish 하지 않음
    # ─────────────────────────────────────────────
    def _check_keywords(self, text: str):
        for kw in PAUSE_KEYWORDS:
            if kw in text:
                if not self._paused:
                    self._paused = True
                    self._publish(True)
                    self.get_logger().warn(f"🎤 정지 키워드 감지: \"{kw}\"")
                return

        for kw in RESUME_KEYWORDS:
            if kw in text:
                if self._paused:
                    self._paused = False
                    self._publish(False)
                    self.get_logger().info(f"🎤 재개 키워드 감지: \"{kw}\"")
                return

    def destroy_node(self):
        self._stop_event.set()
        self._thread.join(timeout=5)
        super().destroy_node()


# ─────────────────────────────────────────────
# ROS2 실행 entry point
# ─────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = VoiceDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
