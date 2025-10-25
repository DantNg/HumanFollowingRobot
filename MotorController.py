import serial

class MotorController:
    def __init__(self, port="COM16", baudrate=115200, timeout=0.1):
        try:
            self.ser = serial.Serial(port, baudrate, timeout=timeout)
            print(f"[MotorController] Connected {port}")
        except Exception as e:
            print(f"[MotorController] Error: {e}")
            self.ser = None
        self.last_cmd = None

    def send_motor_command(self, left_speed, left_dir, right_speed, right_dir):
        if self.ser and self.ser.is_open:
            l_cmd = f"L{left_speed}-{left_dir}\r\n"
            r_cmd = f"R{right_speed}-{right_dir}\r\n"
            self.ser.write(l_cmd.encode())
            self.ser.write(r_cmd.encode())
        else:
            print(f"[SIM] L{left_speed}-{left_dir} | R{right_speed}-{right_dir}")

    def send_if_changed(self, L, Ld, R, Rd):
        tup = (L, Ld, R, Rd)
        if tup != self.last_cmd:
            self.send_motor_command(L, Ld, R, Rd)
            self.last_cmd = tup

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()

    def __del__(self):
        self.close()
