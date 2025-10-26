from tracking import HumanTracking
from MotorController import MotorController
from Lidar import Lidar
import cv2
import time
import threading

# --- Helper: atomic string holder ---
class AtomicAction:
    def __init__(self, initial="stop"):
        self._lock = threading.Lock()
        self._action = initial
    def set(self, value: str):
        v = (value or "stop").strip().lower()
        with self._lock:
            self._action = v
    def get(self) -> str:
        with self._lock:
            return self._action
import time

class ActionArbiter:
    """
    Chống ping-pong giữa LiDAR và Camera bằng lock + cooldown + no-flip.
    """
    def __init__(self,
                 lock_ms=800,           # thời gian bắt buộc giữ hướng tránh sau khi LiDAR yêu cầu
                 cooldown_ms=1000,      # sau khi LiDAR báo 'forward', chỉ cho forward 1 lúc
                 need_strong_bias=0.60, # |x_norm| vượt ngưỡng này mới cho phép phá cooldown
                 need_frames=4):        # số frame liên tiếp vượt ngưỡng
        self.lock_ms = lock_ms
        self.cooldown_ms = cooldown_ms
        self.need_strong_bias = need_strong_bias
        self.need_frames = need_frames

        self.mode = "IDLE"             # IDLE | LOCK | COOLDOWN
        self.lock_until = 0.0
        self.cooldown_until = 0.0
        self.last_avoid_dir = None     # 'left' | 'right'
        self.bias_counter = 0          # đếm frame |x_norm| lớn để phá cooldown

    @staticmethod
    def _now_ms():
        return time.time() * 1000.0

    @staticmethod
    def _is_turn(act):
        return act in ("left","right")

    def update(self, lidar_action: str, cam_action: str, x_norm: float) -> str:
        """
        Trả về final_action sau khi áp hysteresis.
        - lidar_action: 'forward'|'left'|'right'|'stop'
        - cam_action:   'forward'|'left'|'right'|'stop'
        - x_norm: độ lệch [-1..1] từ camera (0=center). Dùng để cho phép phá cooldown khi lệch quá mạnh.
        """
        now = self._now_ms()
        la = (lidar_action or "stop").lower()
        ca = (cam_action  or "stop").lower()

        # 1) Nếu LiDAR yêu cầu né/dừng → vào LOCK
        if la in ("left","right","stop"):
            self.mode = "LOCK"
            self.lock_until = now + self.lock_ms
            self.last_avoid_dir = la if la in ("left","right") else self.last_avoid_dir
            self.bias_counter = 0
            return la

        # 2) LiDAR 'forward' (clear)
        if self.mode == "LOCK":
            # còn trong thời gian LOCK?
            if now < self.lock_until:
                # vẫn giữ hướng tránh nếu có, hoặc forward nếu vừa 'stop'
                return self.last_avoid_dir if self.last_avoid_dir else "forward"
            # hết LOCK → sang COOLDOWN
            self.mode = "COOLDOWN"
            self.cooldown_until = now + self.cooldown_ms
            self.bias_counter = 0

        if self.mode == "COOLDOWN":
            # Cho phá cooldown nếu lệch rất mạnh liên tục need_frames
            if abs(x_norm) > self.need_strong_bias:
                self.bias_counter += 1
            else:
                self.bias_counter = 0

            if self.bias_counter >= self.need_frames:
                # cho camera giành quyền nhưng KHÔNG được đảo ngược hướng tránh ngay
                if self.last_avoid_dir == "left" and ca == "right":
                    return "forward"        # chặn đảo chiều ngược hẳn
                if self.last_avoid_dir == "right" and ca == "left":
                    return "forward"
                return ca  # cùng chiều hoặc forward

            # chưa đủ mạnh → ép forward trong cooldown
            if now < self.cooldown_until:
                return "forward"
            else:
                # hết cooldown → trở về IDLE
                self.mode = "IDLE"

        # 3) IDLE: LiDAR clear, cho camera quyết
        return ca

class FollowRobot:
    def __init__(self, cam_index=0, motor_port="COM20", motor_baud=115200,
                 lidar_port="COM15", lidar_baud=115200):
        self.arbiter = ActionArbiter(
            lock_ms=800,
            cooldown_ms=1000,
            need_strong_bias=0.60,
            need_frames=4
        )
        # Camera → hướng (tracking chịu trách nhiệm)
        self.tracker = HumanTracking()

        # Motor UART
        self.motor = MotorController(port=motor_port, baudrate=motor_baud)

        # LiDAR → phanh & né
        # Lidar.get_full_rotation_dict(...) nên trả về dict angle->distance_mm
        # và decide_direction(scan) trả về: "forward" | "left" | "right" | "stop" | "clear"
        self.lidar = Lidar(port=lidar_port, baud=lidar_baud, has_intensity=False, model="triangle")

        # Camera
        self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Shared actions
        self.cam_action = AtomicAction("stop")
        self.lidar_action = AtomicAction("stop")

        self._stop_event = threading.Event()

        # Speed caps
        self.MAX_SPEED_FORWARD = 20   # % PWM cho tiến
        self.MAX_SPEED_TURN    = 12   # % PWM cho quay/trái/phải

    # -------------- Threads --------------
    def camera_thread(self):
        try:
            while not self._stop_event.is_set():
                ok, frame = self.cap.read()
                if not ok:
                    self.cam_action.set("stop")
                    time.sleep(0.05)
                    continue

                processed = self.tracker.process_frame(frame)
                action = self.tracker.state.get("action", "stop")  # already normalized in tracking
                self.cam_action.set(action)

                # (Tùy chọn) hiển thị
                cv2.imshow("Camera Dir (Face/Head Proxy)", processed)
                if cv2.waitKey(1) & 0xFF == 27:
                    self._stop_event.set()
                    break
        finally:
            try:
                self.cap.release()
            except:
                pass
            cv2.destroyAllWindows()

    def lidar_thread(self):
        from Robot import decide_direction  # đảm bảo trả về lowercase action/clear
        try:
            while not self._stop_event.is_set():
                scan = self.lidar.get_full_rotation_dict(
                    total_timeout=0.5,
                    min_coverage_deg=180.0,
                    wrap_hysteresis_deg=20.0,
                    angle_round=1.0,
                    prefer="min",
                    max_range_mm=3000.0
                )
                if scan is None:
                    # Không có dữ liệu → an toàn: STOP
                    action = "stop"
                else:
                    action = decide_direction(scan) or "forward"
                    action = action.strip().lower()
                    # chuẩn hoá: cho phép 'clear' nghĩa là an toàn → xem như forward (không cản trở)
                    if action == "clear":
                        action = "forward"

                self.lidar_action.set(action)
                time.sleep(0.03)
        finally:
            try:
                self.lidar.close()
            except:
                pass

    # -------------- Motor mapping --------------
    def _apply_motor_action(self, final_action: str, speed_hint_pct: int):
        """
        final_action: 'forward' | 'left' | 'right' | 'stop'
        speed_hint_pct: 0..50 (từ tracking) — dùng để scale mềm, vẫn clamp bởi MAX_SPEED_*
        """
        # Scale gợi ý từ camera (0..50) → (0..cap)
        fwd = 20#max(0, min(self.MAX_SPEED_FORWARD, int((speed_hint_pct/50.0)*self.MAX_SPEED_FORWARD)))
        turn = 10#max(0, min(self.MAX_SPEED_TURN,    int((speed_hint_pct/50.0)*self.MAX_SPEED_TURN)))

        if final_action == "forward":
            L = R = fwd
            now = time.time()
            post_until = getattr(self, "_post_turn_until", 0.0)
            # Nếu đang ở giai đoạn "sau quay -> đi thẳng 2s", ép về forward
            if now < post_until:
                L = R = fwd
                self.motor.send_if_changed(L, 1, R, 1)
                return

            if final_action == "forward":
                self.motor.send_if_changed(L, 1, R, 1)

            elif final_action == "left":
                # Quay/trườn trái: bánh phải nhanh hơn bánh trái (không lùi bánh trái)
                L = max(0, int(0.25*turn))
                R = turn
                self.motor.send_if_changed(L, 1, R, 1)
                # Sau khi bắt đầu quay, chuyển sang đi thẳng trong 2 giây
                self._post_turn_until = time.time() + 2.0

            elif final_action == "right":
                # Quay/trườn phải: bánh trái nhanh hơn bánh phải (không lùi bánh phải)
                L = turn
                R = max(0, int(0.25*turn))
                self.motor.send_if_changed(L, 1, R, 1)
                # Sau khi bắt đầu quay, chuyển sang đi thẳng trong 2 giây
                self._post_turn_until = time.time() + 2.0

            else:  # "stop" hoặc bất kỳ
                self._post_turn_until = 0.0
                self.motor.send_if_changed(0, 1, 0, 1)

        else:  # "stop" hoặc bất kỳ
            self.motor.send_if_changed(0, 1, 0, 1)

    # -------------- Fusion policy --------------
    @staticmethod
    def _fuse_actions(cam_action: str, lidar_action: str) -> str:
        """
        Quy tắc:
          - Nếu LiDAR an toàn (forward): theo camera (cam_action).
          - Nếu LiDAR yêu cầu né/dừng (left/right/stop): ghi đè LiDAR.
        """
        la = (lidar_action or "stop").lower()
        ca = (cam_action or "stop").lower()
        if la == "forward":   # clear
            return ca
        # ưu tiên an toàn
        if la in ("left", "right", "stop"):
            return la
        # fallback
        return ca

    # -------------- Main loop --------------
    def run(self):
        t_cam = threading.Thread(target=self.camera_thread, daemon=True)
        t_lid = threading.Thread(target=self.lidar_thread, daemon=True)
        t_cam.start()
        t_lid.start()

        try:
            while not self._stop_event.is_set():
                cam_action   = self.cam_action.get()      # 'forward'|'left'|'right'|'stop'
                lidar_action = self.lidar_action.get()    # 'forward'|'left'|'right'|'stop'
                speed_pct    = int(self.tracker.state.get("speed_pct", 0))  # nếu tracking có, không bắt buộc

                x_norm = float(self.tracker.state.get("x_smooth", 0.0))
                final_action = self.arbiter.update(lidar_action, cam_action, x_norm)

                # Log gọn
                print(f"[FUSE] cam={cam_action:>7}  lidar={lidar_action:>7}  => final={final_action:>7}  speed_hint={speed_pct}")

                self._apply_motor_action(final_action, speed_pct)
                time.sleep(0.03)

        except KeyboardInterrupt:
            pass
        finally:
            self._stop_event.set()
            try:
                t_cam.join(timeout=1.0)
                t_lid.join(timeout=1.0)
            except:
                pass
            # đảm bảo dừng
            self.motor.send_if_changed(0, 1, 0, 1)

def main():
    robot = FollowRobot()
    robot.run()

if __name__ == "__main__":
    main()
