import logging

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from config import YOLO_MODEL_PATH
from ros.cam_state import cam_state, cam_lock

log = logging.getLogger(__name__)

class CameraNode(Node):
    """
    RealSense 카메라 토픽을 구독해 컬러/깊이 프레임을 cam_state에 저장하는 ROS 노드.

    구독 토픽:
        /camera/camera/color/image_raw        — 컬러 이미지 (BGR)
        /camera/camera/aligned_depth_to_color/image_raw — 깊이 이미지 (mm, uint16)
        /camera/camera/color/camera_info      — 카메라 내부 파라미터

    Note:
        ROS 콜백(_color_cb 등)은 spin 스레드에서 호출되므로
        무거운 연산(YOLO 추론)은 yolo_on 플래그 확인 후에만 실행한다.
        콜백이 블록되면 이후 메시지가 큐에 쌓여 지연이 누적되기 때문.
    """

    def __init__(self) -> None:
        super().__init__("lego_admin_ui_cam")
        self._bridge: CvBridge = CvBridge()
        self._yolo_model = None  # 최초 필요 시점에 lazy load

        # RELIABLE+VOLATILE: 데이터 손실 없이 최신 프레임 수신
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.create_subscription(
            Image,
            "/camera/camera/color/image_raw",
            self._color_cb,
            qos,
        )
        self.create_subscription(
            Image,
            "/camera/camera/aligned_depth_to_color/image_raw",
            self._depth_cb,
            qos,
        )
        self.create_subscription(
            CameraInfo,
            "/camera/camera/color/camera_info",
            self._cam_info_cb,
            qos,
        )
        self.get_logger().info("Camera subscriber ready")
        log.info("[CAM] CameraNode 구독 등록 완료")

    def _color_cb(self, msg: Image) -> None:
        """
        컬러 이미지 콜백. YOLO 추론 및 프레임 JPEG 인코딩을 수행한다.

        처리 흐름:
        1. ROS Image → BGR ndarray 변환
        2. yolo_on이면 lazy load 후 추론 → 탐지 결과 cam_state 갱신
        3. 640×480 리사이즈 후 JPEG 인코딩 → cam_state["frame"] 갱신

        Note:
            YOLO 추론은 콜백 내에서 동기 실행되므로
            프레임율이 모델 추론 속도에 의존한다 (conf=0.90으로 낮은 오탐 억제)
        """
        from ultralytics import YOLO as _YOLO

        frame: np.ndarray = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        with cam_lock:
            yolo_on: bool = cam_state["yolo_on"]

        if yolo_on:
            # 최초 yolo_on 시점에만 모델을 로드해 메모리 낭비 방지
            if self._yolo_model is None:
                log.info("[YOLO] 모델 로드 시작: %s", YOLO_MODEL_PATH)
                try:
                    self._yolo_model = _YOLO(YOLO_MODEL_PATH)
                    log.info("[YOLO] 모델 로드 완료")
                except Exception as e:
                    log.error("[YOLO] 모델 로드 실패: %s", e)

            if self._yolo_model is not None:
                results = self._yolo_model.predict(source=frame, conf=0.80, verbose=False)
                result  = results[0]
                obbs    = result.obb
                classes = result.names
                detections: list[dict] = []

                if obbs is not None:
                    for obb in obbs:
                        cls_id: int   = int(obb.cls[0])
                        conf: float   = float(obb.conf[0]) * 100  # 0~100 %
                        cls_full: str = classes[cls_id]
                        color: str    = cls_full.split('_')[0]

                        pts   = obb.xyxyxyxy[0].cpu().numpy().astype(int)
                        xywhr = obb.xywhr[0].cpu().numpy()

                        det_cx: int    = int(xywhr[0])    # 픽셀 좌표
                        det_cy: int    = int(xywhr[1])    # 픽셀 좌표
                        w_obb: float   = float(xywhr[2])  # OBB 너비 (픽셀)
                        h_obb: float   = float(xywhr[3])  # OBB 높이 (픽셀)
                        angle_deg: float = float(np.degrees(float(xywhr[4])))
              
                        # YOLO OBB 각도는 -90~90°; 로봇 제어계와 맞추기 위해 +90° 보정
                        angle_deg += 90   
                        angle_deg = (angle_deg + 90.0) % 180.0 - 90.0
                        # 종횡비 1.3 미만은 2x2(정사각형에 가까움)로 판별
                        ratio: float = (
                            max(w_obb, h_obb) / min(w_obb, h_obb)
                            if min(w_obb, h_obb) > 0 else 1.0
                        )
                        shape: str = "2x2" if ratio < 1.3 else "2x3"

                        detections.append({
                            "cx": det_cx, "cy": det_cy,
                            "angle_deg": angle_deg,
                            "conf": conf,
                            "color": color, "shape": shape,
                            "pts": pts.tolist(),
                        })

                        cv2.polylines(frame, [pts.reshape((-1, 1, 2))], True, (0, 255, 0), 2)
                        x0, y0 = pts[0]
                        cv2.putText(
                            frame, f"{color} {shape} {conf:.0f}%",
                            (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                        )

                with cam_lock:
                    cam_state["detections"] = detections

        # 스트리밍 부하를 낮추기 위해 640×480으로 다운스케일 후 인코딩
        frame = cv2.resize(frame, (640, 480))
        _, buf = cv2.imencode(".jpg", frame)
        with cam_lock:
            cam_state["frame"] = buf.tobytes()

    def _depth_cb(self, msg: Image) -> None:
        """
        깊이 이미지 콜백. mm 단위 uint16 배열을 cam_state에 저장한다.

        Note:
            aligned_depth_to_color 토픽이므로 컬러 픽셀 좌표와 1:1 대응된다.
        """
        depth: np.ndarray = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        with cam_lock:
            # copy()로 ROS 버퍼 해제 후에도 안전하게 참조 가능
            cam_state["depth_frame"] = depth.copy()

    def _cam_info_cb(self, msg: CameraInfo) -> None:
        """
        카메라 내부 파라미터 콜백. 픽셀 → 3D 역투영에 사용하는 fx,fy,ppx,ppy를 저장한다.

        Note:
            K 행렬 인덱스: K[0]=fx, K[4]=fy, K[2]=ppx, K[5]=ppy
        """
        with cam_lock:
            cam_state["cam_info"] = {
                "fx":  msg.k[0],   # 초점거리 x (픽셀)
                "fy":  msg.k[4],   # 초점거리 y (픽셀)
                "ppx": msg.k[2],   # 주점 x (픽셀)
                "ppy": msg.k[5],   # 주점 y (픽셀)
            }
