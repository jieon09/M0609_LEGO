#!/usr/bin/env python3
from pymodbus.client.sync import ModbusTcpClient as ModbusClient


class RG:
    """OnRobot RG2/RG6 그리퍼 Modbus TCP 제어."""

    def __init__(self, gripper: str, ip: str, port: int) -> None:
        self.client = ModbusClient(ip, port=port, stopbits=1, bytesize=8, parity="E", baudrate=115200, timeout=1)
        if gripper not in ["rg2", "rg6"]:
            print("Please specify either rg2 or rg6.")
            return
        self.gripper = gripper
        if self.gripper == "rg2":
            self.max_width = 1100
            self.max_force = 400
        elif self.gripper == "rg6":
            self.max_width = 1600
            self.max_force = 1200
        self.open_connection()

    def open_connection(self) -> None:
        self.client.connect()

    def close_connection(self) -> None:
        self.client.close()

    def move_gripper(self, width_01mm: int, force_val: int = 400) -> None:
        params = [force_val, width_01mm, 16]
        print(f"Moving gripper to {width_01mm * 0.1} mm.")
        self.client.write_registers(address=0, values=params, unit=65)

    def close_gripper(self, force_val: int = 400) -> None:
        self.move_gripper(0, force_val)

    def open_gripper(self, force_val: int = 400) -> None:
        self.move_gripper(self.max_width, force_val)
