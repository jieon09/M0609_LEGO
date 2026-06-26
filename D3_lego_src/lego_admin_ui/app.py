import logging

from dotenv import load_dotenv
load_dotenv()

log = logging.getLogger(__name__)

from flask import Flask, Response, render_template

# 다른 모듈 import 전에 로깅을 먼저 설정해야 ROS/SQLAlchemy 초기 로그도 포맷 적용됨
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
# werkzeug 자체 요청 로그는 너무 잦아 WARNING 이상만 허용
logging.getLogger("werkzeug").setLevel(logging.WARNING)

from api.task_api import task_bp
from api.robot_api import robot_bp
from api.camera_api import camera_bp
from api.action_api import action_bp
from ros.ros_thread import start_camera_thread
from services.automation_service import start_automation_loop

app = Flask(__name__)
app.register_blueprint(task_bp)
app.register_blueprint(robot_bp)
app.register_blueprint(camera_bp)
app.register_blueprint(action_bp)


@app.route("/")
def index() -> Response:
    """관리자 대시보드 메인 페이지를 반환한다."""
    return render_template("index.html")


# 앱 시작 시 백그라운드 스레드를 일괄 기동 (daemon=True → 메인 프로세스 종료 시 자동 회수)
start_camera_thread()
start_automation_loop()


if __name__ == "__main__":
    import socket
    host_ip = socket.gethostbyname(socket.gethostname())
    log.info("관리자 UI 실행 중")
    log.info("Local:   http://127.0.0.1:7000/")
    log.info("Network: http://%s:7000/", host_ip)
    app.run(
        host="0.0.0.0",
        port=7000,
        debug=False,   # host=0.0.0.0 노출 환경에서 Werkzeug debugger는 RCE 위험
        use_reloader=False,  # 리로더가 켜지면 백그라운드 스레드가 이중 기동됨
        threaded=True,
    )
