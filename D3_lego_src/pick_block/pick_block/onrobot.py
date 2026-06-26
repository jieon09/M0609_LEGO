from pymodbus.client.sync import ModbusTcpClient as ModbusClient


class RG:
    """OnRobot RG2 / RG6 gripper control via Modbus TCP."""

    def __init__(self, gripper: str, ip: str, port: int) -> None:
        """
        RG 그리퍼 초기화 및 Modbus TCP 연결

        Args:
            gripper: 그리퍼 종류 ("rg2" 또는 "rg6")
            ip: 툴체인저 IP 주소
            port: Modbus TCP 포트 번호

        Raises:
            ValueError: gripper가 rg2/rg6 이외의 값인 경우
        """
        self.client = ModbusClient(
            ip,
            port=port,
            stopbits=1,
            bytesize=8,
            parity="E",
            baudrate=115200,
            timeout=1,
        )

        if gripper not in ["rg2", "rg6"]:
            raise ValueError("gripper must be rg2 or rg6")

        self.gripper: str = gripper
        # 너비 단위: 0.1mm, 힘 단위: 0.1N (OnRobot Modbus 레지스터 스케일)
        self.max_width: int = 700 if gripper == "rg2" else 1600
        self.max_force: int = 400 if gripper == "rg2" else 1200

        self.open_connection()

    def open_connection(self) -> None:
        """Modbus TCP 연결을 수립한다."""
        self.client.connect()

    def close_connection(self) -> None:
        """Modbus TCP 연결을 해제한다."""
        self.client.close()

    def open_gripper(self, force_val: int = 400) -> None:
        """
        그리퍼를 최대 너비로 열기

        Args:
            force_val: 파지력 (0.1N 단위, 열 때 반력 기준값으로 사용됨)

        Note:
            Modbus 레지스터 0번지에 [force, width, control] 순으로 기록.
            unit=65는 OnRobot 툴체인저 Modbus slave ID이다.
        """
        self.client.write_registers(
            address=0,
            values=[force_val, self.max_width, 16],
            unit=65,
        )
        print("Gripper Open")

    def close_gripper(self, force_val: int = 400) -> None:
        """
        그리퍼를 닫기 (너비=0)

        Args:
            force_val: 파지력 (0.1N 단위, RG2 기준 최대 400)

        Note:
            Modbus 레지스터 0번지에 [force, width=0, control] 순으로 기록.
            unit=65는 OnRobot 툴체인저 Modbus slave ID이다.
        """
        self.client.write_registers(
            address=0,
            values=[force_val, 0, 16],
            unit=65,
        )
        print("Gripper Close")
