# LEGO Robot UI (Admin + Customer)

---

## 1. 주요 기능

### 고객 UI (`lego_customer_ui/app.py`, port **5000**)
- **실시간 카메라 피드**: USB 카메라 MJPEG 스트리밍 (`/video_feed`, `/video_feed_hsv`, `/video_feed_debug`)
- **그림 캡처 & 종이 윤곽 보정**: 사용자가 그린 그림을 캡처하여 원근 보정 후 저장 (`/capture`)
- **주문 등록**: 캡처된 이미지로 Task 생성 → 객체 분할(`object_segment_fin.py`) → 레고 변환(`lego_convert_fin.py`) 파이프라인 자동 실행 (`/start_draw`)
- **객체 다중 선택 UI**: 분할된 객체가 여러 개면 고객이 직접 선택 (`/object_selection_status`, `/confirm_object_selection`)
- **재고 / 작업 현황 조회**: 레고 재고 및 최근 작업 목록 표시 (`/inventory`, `/task_status`, `/build_status/<id>`)

### 관리자 UI (`lego_admin_ui/app.py`, port **7000**)
- **로봇 제어 API** (`api/`): Task, Robot, Camera, Action 블루프린트
- **ROS2 노드 연동** (`ros/`): CameraNode, HandNode, VoiceNode → `MultiThreadedExecutor`로 spin
- **자동화 루프** (`services/automation_service.py`): pick → place → home 자동 수행, 복구/로그 서비스 포함
- **TTS 음성 안내** (`voice/voice_tts.py`): gTTS 기반 안내 음성 재생

---

## 2. 시스템 설계 플로우 차트

```
 ┌────────────────────┐         ┌────────────────────┐
 │   고객 UI (5000)    │         │  관리자 UI (7000)   │
 │  lego_customer_ui  │         │   lego_admin_ui    │
 └─────────┬──────────┘         └─────────┬──────────┘
           │                              │
           │  ① 그림 캡처/주문 등록         │ ② ROS2 노드 spin
           │  (Task 생성)                  │ (Camera/Hand/Voice)
           ▼                              ▼
 ┌──────────────────────────────────────────────────┐
 │           PostgreSQL  (lego_robot DB)            │
 │   Task / Inventory / RobotLog / RobotAction      │
 └──────────────────────────────────────────────────┘
           ▲                              ▲
           │ ④ brick_path 저장             │ ③ 자동화 루프가
           │ (변환 결과)                   │   Task 폴링
           │                              │
 ┌─────────┴──────────┐         ┌─────────┴──────────┐
 │ object_segment_fin │         │ automation_service │
 │ lego_convert_fin   │         │ pick / place / home│
 │ (subprocess)       │         │ (로봇 제어)         │
 └────────────────────┘         └────────────────────┘
```

---

## 3. 운영체제 환경

| 항목 | 버전 |
|------|------|
| OS | Ubuntu 22.04 LTS (Linux 6.8) |
| Python | 3.10+ |
| ROS2 | Humble |
| DB | PostgreSQL 14+ |
| Shell | bash |

---

## 4. 사용한 장비 목록

| 장비 | 용도 |
|------|------|
| 협동 로봇 암 (Cobot) | 레고 블록 pick & place |
| USB 웹캠 | 고객 그림 캡처 / 작업 모니터링 |
| RGB-D 카메라 (ROS2 `sensor_msgs/Image`, `CameraInfo`) | 로봇 비전 (관리자 UI 연동) |
| 손 감지 센서 / 노드 (`std_msgs/Bool`) | 안전 정지용 손 진입 감지 |
| 스피커 | gTTS 음성 안내 출력 |
| PostgreSQL 서버 | Task / Inventory / Log 저장 |

---

## 5. 의존성 목록

### Python 패키지 (공통)
```
flask>=3.0
python-dotenv
sqlalchemy
psycopg2-binary
opencv-python
numpy
Pillow>=10.0
gTTS
```

### ROS2 (관리자 UI 전용)
```
rclpy
sensor_msgs
std_msgs
cv_bridge
```

### 외부 시스템
- PostgreSQL (`postgresql://postgres:1234@localhost:5432/lego_robot`)
- ROS2 Humble runtime (`source /opt/ros/humble/setup.bash`)

---

## 6. 실행 순서

### ① PostgreSQL 기동 & DB 생성
```bash
sudo service postgresql start
sudo -u postgres psql -c "CREATE DATABASE lego_robot;"
```

### ② `.env` 확인 (양쪽 디렉터리 모두)
```
DATABASE_URL=postgresql://postgres:1234@localhost:5432/lego_robot
```
파일 위치:
- [lego_admin_ui/.env](/.env)
- [lego_customer_ui/.env](../lego_customer_ui/.env)

### ③ 의존성 설치
```bash
# 관리자 UI
cd ~/cobot_ws/src/lego_admin_ui
pip install -r requirements.txt
pip install python-dotenv sqlalchemy psycopg2-binary opencv-python numpy gTTS

# 고객 UI (requirements.txt 없음 → 동일 패키지 설치)
cd ~/cobot_ws/src/lego_customer_ui
pip install flask python-dotenv sqlalchemy psycopg2-binary opencv-python numpy gTTS Pillow

# 손 감지(MediaPipe) / 음성 인식(STT) — 별도 Docker 컨테이너로 실행
# (로컬 pip 설치 불필요, 아래 ⑤-1 단계 참고)
```

### ④ DB 테이블 초기화 (최초 1회)
```bash
cd ~/cobot_ws/src/lego_admin_ui
python database/init_db.py
```
→ Task / Inventory / RobotLog 테이블 생성 + 재고 초기 행 삽입

### ⑤ ROS2 환경 source (관리자 UI 실행 셸에서)
```bash
source /opt/ros/humble/setup.bash
source ~/cobot_ws/install/setup.bash   # 워크스페이스 빌드 후
```

### ⑤-1 손 감지 / 음성 인식 Docker 컨테이너 기동
```bash
# 손 감지 (MediaPipe) — std_msgs/Bool 토픽 publish
docker run --rm -it --network host --device /dev/video0 <hand-detect-image>

# 음성 인식 (STT) — std_msgs/Bool 토픽 publish
docker run --rm -it --network host --device /dev/snd  <stt-image>
```
→ 고객 UI에서 컨테이너가 publish하는 토픽을 관리자 UI의 `HandNode` / `VoiceNode`가 구독함

### ⑥ 관리자 UI 실행 (터미널 1)

먼저 RealSense 카메라와 두산 로봇 bringup을 기동한다 (`~/.bashrc` alias 사용).

```bash
# RealSense D-시리즈 카메라 (depth + RGB + pointcloud)
realsense
# == ros2 launch realsense2_camera rs_align_depth_launch.py \
#      depth_module.depth_profile:=848x480x30 \
#      rgb_camera.color_profile:=1280x720x30 \
#      initial_reset:=true \
#      align_depth.enable:=true \
#      enable_rgbd:=true \
#      pointcloud.enable:=true

# 두산 M0609 로봇 bringup (실기 모드)
roboton
# == ros2 launch dsr_bringup2 dsr_bringup2_rviz.launch.py \
#      mode:=real host:=192.168.1.100 port:=12345 model:=m0609
#
```

그 다음 관리자 UI 실행:

```bash
cd ~/cobot_ws/src/lego_admin_ui
python app.py
# → http://127.0.0.1:7000/
```
실행 시 자동 기동:
- `start_camera_thread()` — ROS2 Camera/Hand/Voice 노드 spin
- `start_automation_loop()` — pick/place 자동화 루프


### ⑦ 고객 UI 실행 (터미널 2)
```bash
cd ~/cobot_ws/src/lego_customer_ui
python app.py
# → http://127.0.0.1:5000/
```

### ⑧ 브라우저 접속
| UI | URL |
|----|-----|
| 고객 UI | http://localhost:5000 |
| 관리자 UI | http://localhost:7000 |

> **주의**
> - 관리자 UI는 `debug=False`, `use_reloader=False`로 고정 — 리로더가 켜지면 ROS2 노드가 이중 기동됨
> - 고객 UI는 카메라를 점유하므로 동일 USB 카메라를 다른 프로세스가 사용 중이면 안 됨
> - 두 UI 모두 동일 PostgreSQL DB(`lego_robot`)를 공유함
