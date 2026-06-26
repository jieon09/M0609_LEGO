import logging
from typing import Generator

from flask import Blueprint, Response, jsonify, request

from database.connection import get_db
from database.models import Task, RobotLog

log = logging.getLogger(__name__)

task_bp = Blueprint("task", __name__)


@task_bp.route("/api/tasks")
def api_tasks() -> Response:
    """
    전체 Task 목록을 생성 시각 역순으로 반환한다.

    처리 흐름:
    1. Task 테이블 전체 조회 (최신순)
    2. 각 행을 상태 문자열로 변환 후 직렬화

    Returns:
        Task 배열 JSON — status는 "done" | "ready" | "processing"

    Note:
        brick_path 미존재 → "processing", end_at 미존재 → "ready",
        end_at 존재 → "done"
    """
    log.debug("[API] GET /api/tasks 요청 수신")
    db = next(get_db())
    try:
        customers: list[Task] = db.query(Task).order_by(Task.created_at.desc()).all()
        log.debug("[API] /api/tasks 조회 결과: %d건", len(customers))
        return jsonify([
            {
                "id":          str(c.id),
                "customer_id": c.customer_id,
                "status":      "done" if c.end_at else ("ready" if c.brick_path else "processing"),
                "img_path":    c.img_path,
                "created_at":  c.created_at.isoformat() if c.created_at else None,
                "end_at":      c.end_at.isoformat()     if c.end_at     else None,
            }
            for c in customers
        ])
    finally:
        db.close()


@task_bp.route("/api/robot_logs")
def api_robot_logs() -> Response:
    """
    로봇 동작 로그를 최신순으로 최대 100건 반환한다.

    처리 흐름:
    1. task_id 쿼리 파라미터가 있으면 해당 Task 로그만 필터링
    2. 최신순 정렬 후 100건 제한

    Args (query param):
        task_id (optional): 특정 Task UUID로 필터링

    Returns:
        RobotLog 배열 JSON — joints (deg), tcp (mm / deg, robot base 좌표계)
    """
    task_id: str | None = request.args.get("task_id")
    log.debug("[API] GET /api/robot_logs 요청 수신 (task_id=%s)", task_id)
    db = next(get_db())
    try:
        q = db.query(RobotLog).order_by(RobotLog.created_at.desc())
        if task_id:
            q = q.filter(RobotLog.task_id == task_id)
        logs: list[RobotLog] = q.limit(100).all()
        return jsonify([
            {
                "log_id":     str(entry.log_id),
                "task_id":    str(entry.task_id),
                "status":     entry.robot_action.value,
                "created_at": entry.created_at.isoformat() if entry.created_at else None,
                "joints": {
                    "j1": entry.j1, "j2": entry.j2, "j3": entry.j3,
                    "j4": entry.j4, "j5": entry.j5, "j6": entry.j6,
                },
                "tcp": {
                    "x": entry.tcp_x,  "y": entry.tcp_y,  "z": entry.tcp_z,
                    "rx": entry.tcp_rx, "ry": entry.tcp_ry, "rz": entry.tcp_rz,
                },
            }
            for entry in logs
        ])
    finally:
        db.close()
