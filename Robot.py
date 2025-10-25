from Lidar import Lidar

from MotorController import MotorController

def decide_direction(scan, left_range=(160, 179), right_range=(181, 200), obstacle_thresh=500):
    """
    scan: dict {angle: distance}
    left_range: tuple (start_deg, end_deg) for left sector
    right_range: tuple (start_deg, end_deg) for right sector
    obstacle_thresh: mm, dưới ngưỡng này coi là có vật cản
    Return: 'forward', 'left', 'right'
    """
    left_blocked = False
    right_blocked = False
    # Kiểm tra bên trái
    for ang in range(left_range[0], left_range[1]+1):
        d = scan.get(float(ang))
        if d is not None and d > 0 and d < obstacle_thresh:
            left_blocked = True
            break
    # Kiểm tra bên phải
    for ang in range(right_range[0], right_range[1]+1):
        d = scan.get(float(ang))
        if d is not None and d > 0 and d < obstacle_thresh:
            right_blocked = True
            break
    # Quyết định hướng
    if left_blocked and right_blocked:
        return 'stop'  # Cả 2 bên đều bị chặn (có thể xử lý đặc biệt)
    elif left_blocked:
        return 'right'
    elif right_blocked:
        return 'left'
    else:
        return 'forward'
    

import threading
import time

class Robot:
	def __init__(self, port="COM15", baud=115200, has_intensity=False, model="triangle", io_timeout=0.2):
		self.lidar = Lidar(port, baud, has_intensity=has_intensity, model=model, io_timeout=io_timeout)
		self.motor_controller = MotorController(port="COM19", baudrate=115200, timeout=0.1)
		self.latest_scan = None
		self._stop_thread = False
		self._scan_lock = threading.Lock()
		self._thread = threading.Thread(target=self._scan_loop, daemon=True)

	def start_lidar(self):
		self._stop_thread = False
		if not self._thread.is_alive():
			self._thread = threading.Thread(target=self._scan_loop, daemon=True)
			self._thread.start()

	def stop_lidar(self):
		self._stop_thread = True
		if self._thread.is_alive():
			self._thread.join()

	def _scan_loop(self):
		with self.lidar as ld:
			while not self._stop_thread:
				scan = ld.get_full_rotation_dict(
					total_timeout=0.5,  # giảm timeout để cập nhật nhanh hơn
					min_coverage_deg=180.0,  # chấp nhận độ phủ nhỏ hơn
					wrap_hysteresis_deg=20.0,
					angle_round=1.0,
					prefer="min",
					max_range_mm=3000.0
				)
				with self._scan_lock:
					self.latest_scan = scan

	def get_latest_scan(self):
		with self._scan_lock:
			return self.latest_scan.copy() if self.latest_scan else None

	def forward(self):
		print("FORWARD")
		self.motor_controller.send_if_changed(10, 1, 10, 1)

	def left(self):
		print("LEFT")
		self.motor_controller.send_if_changed(10, 0, 10, 1)

	def right(self):
		print("RIGHT")
		self.motor_controller.send_if_changed(10, 1, 10, 0)

	def stop(self):
		print("STOP")
		self.motor_controller.send_if_changed(0, 1, 0, 1)

	def run(self):
		self.start_lidar()
		try:
			while True:
				scan = self.get_latest_scan()
				if scan is None:
					print("❌ Chưa có dữ liệu LiDAR")
					self.stop()
					time.sleep(0.05)
					continue
				direction = decide_direction(scan)
				if direction == 'forward':
					self.forward()
				elif direction == 'left':
					self.left()
				elif direction == 'right':
					self.right()
				else:
					self.stop()
				time.sleep(0.05)
		finally:
			self.stop_lidar()

# Demo sử dụng
if __name__ == "__main__":
	robot = Robot()
	robot.run()

