#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-05-01
# @Filename: test_dome.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from time import sleep

from pymodbus.bit_read_message import ReadCoilsResponse as RCR
from pymodbus.client import ModbusTcpClient
from pymodbus.register_read_message import ReadHoldingRegistersResponse as RHRC


ADDRESS = "10.8.38.51"
DIRECTION = True


def report_variables(client: ModbusTcpClient):
    interlock_state = client.read_coils(2, 1)
    erelay_state = client.read_coils(4, 1)
    drive_enable = client.read_coils(99, 1)
    drive_state = client.read_coils(100, 1)
    drive_direction = client.read_coils(101, 1)
    drive_brake = client.read_coils(102, 1)
    drive_velocity = client.read_holding_registers(103, 2)
    drive_overcurrent = client.read_coils(109, 1)

    assert isinstance(interlock_state, RCR) and not interlock_state.isError()
    assert isinstance(erelay_state, RCR) and not erelay_state.isError()
    assert isinstance(drive_enable, RCR) and not drive_enable.isError()
    assert isinstance(drive_state, RCR) and not drive_state.isError()
    assert isinstance(drive_direction, RCR) and not drive_direction.isError()
    assert isinstance(drive_brake, RCR) and not drive_brake.isError()
    assert isinstance(drive_velocity, RHRC) and not drive_velocity.isError()
    assert isinstance(drive_overcurrent, RCR) and not drive_overcurrent.isError()

    print(f"Interlock state (MC3): {interlock_state.bits[0]}")
    print(f"E-relay state (MC4): {erelay_state.bits[0]}")
    print(f"Drive enable (MC100): {drive_enable.bits[0]}")
    print(f"Drive state (MC101): {drive_state.bits[0]}")
    print(f"Drive direction (MC102): {drive_direction.bits[0]}")
    print(f"Drive brake (MC103): {drive_brake.bits[0]}")
    print(f"Drive velocity (MC104-105): {drive_velocity.registers}")
    print(f"Drive overcurrent (MC110): {drive_overcurrent.bits[0]}")
    print()


def test_dome():
    client = ModbusTcpClient(ADDRESS)
    client.connect()

    report_variables(client)
    sleep(2)
    report_variables(client)

    print("Setting direction and enabling.\n")
    client.write_coil(101, DIRECTION)
    sleep(1)
    client.write_coil(99, True)

    while True:
        report_variables(client)
        sleep(0.5)


if __name__ == "__main__":
    test_dome()
