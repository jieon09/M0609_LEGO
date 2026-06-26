import uuid
import enum
from sqlalchemy import (
    Column, DateTime, Enum, Float, ForeignKey, Integer, String,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database.connection import Base


class RobotAction(str, enum.Enum):
    """로봇 동작 단계 — DB Enum 타입과 1:1 매핑."""
    pick            = "pick"
    place           = "place"
    detect_block    = "detect_block"      # YOLO 블럭 감지
    move_pick_place = "move_pick_place"   # pick 준비 위치 이동
    go_home         = "go_home"           # 홈 위치 복귀


class LogStatus(str, enum.Enum):
    """로봇 로그 행의 처리 상태 열거형."""
    pending   = "pending"    # 작업 대기 중
    running   = "running"    # 실행 중
    success   = "success"    # 정상 완료
    failed    = "failed"     # 실패
    retrying  = "retrying"   # 재시도 중
    emergency = "emergency"  # 긴급 정지 발생


class ErrorCode(str, enum.Enum):
    """로봇 오류 코드 열거형. 원인 분류에 사용한다."""
    PLACE_FAIL    = "PLACE_FAIL"     # 블럭 배치 실패
    VISION_FAIL   = "VISION_FAIL"    # YOLO 탐지 실패
    TIMEOUT       = "TIMEOUT"        # subprocess/동작 timeout
    UNKNOWN_ERROR = "UNKNOWN_ERROR"  # 미분류 오류
    HAND_DETECTED = "HAND_DETECTED"  # 작업 영역 내 손 감지
    VOICE_PAUSED  = "VOICE_PAUSED"   # 음성 명령으로 일시 정지


class Inventory(Base):
    __tablename__ = "inventory"

    inventory_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    red_2x2      = Column(Integer, nullable=False, default=0)
    red_2x3      = Column(Integer, nullable=False, default=0)
    blue_2x2     = Column(Integer, nullable=False, default=0)
    blue_2x3     = Column(Integer, nullable=False, default=0)
    yellow_2x2   = Column(Integer, nullable=False, default=0)
    yellow_2x3   = Column(Integer, nullable=False, default=0)
    total_blocks  = Column(Integer, nullable=False, default=0)

    def __repr__(self) -> str:
        return (
            f"<Inventory red_2x2={self.red_2x2} red_2x3={self.red_2x3} "
            f"blue_2x2={self.blue_2x2} blue_2x3={self.blue_2x3} "
            f"yellow_2x2={self.yellow_2x2} yellow_2x3={self.yellow_2x3} "
            f"total_blocks={self.total_blocks}>"
        )


class Task(Base):
    __tablename__ = "task"

    id          = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    customer_id = Column(String(200), nullable=True)
    brick_path  = Column(JSONB, nullable=True)
    img_path    = Column(String(500), nullable=True)
    created_at  = Column(DateTime(timezone=True), server_default=func.now())
    end_at      = Column(DateTime(timezone=True), nullable=True)

    robot_logs = relationship("RobotLog", back_populates="task", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Task {self.id} | customer={self.customer_id}>"


class RobotLog(Base):
    """로봇 동작 로그 — 각 step별 관절 각도 및 TCP 위치를 기록."""

    __tablename__ = "robot_logs"

    log_id       = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    task_id      = Column(UUID(as_uuid=True), ForeignKey("task.id", ondelete="CASCADE"), nullable=False)
    robot_action = Column(Enum(RobotAction, name="robotaction"), nullable=False)
    step_order   = Column(Integer, nullable=True)
    created_at   = Column(DateTime(timezone=True), server_default=func.now())

    # 처리 상태 및 오류 분류
    status     = Column(Enum(LogStatus,  name="logstatus"),  nullable=True)
    error_code = Column(Enum(ErrorCode,  name="errorcode"),  nullable=True)

    # 관절 각도 (단위: degree)
    j1 = Column(Float, nullable=True)
    j2 = Column(Float, nullable=True)
    j3 = Column(Float, nullable=True)
    j4 = Column(Float, nullable=True)
    j5 = Column(Float, nullable=True)
    j6 = Column(Float, nullable=True)

    # TCP 위치: x/y/z (단위: mm, robot base 좌표계), rx/ry/rz (단위: degree)
    tcp_x  = Column(Float, nullable=True)
    tcp_y  = Column(Float, nullable=True)
    tcp_z  = Column(Float, nullable=True)
    tcp_rx = Column(Float, nullable=True)
    tcp_ry = Column(Float, nullable=True)
    tcp_rz = Column(Float, nullable=True)

    task = relationship("Task", back_populates="robot_logs")

    def __repr__(self) -> str:
        return (
            f"<RobotLog task={self.task_id} step={self.step_order} "
            f"action={self.robot_action} status={self.status} error={self.error_code}>"
        )
