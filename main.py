import cv2
import time
import threading
import math
from ultralytics import YOLO

from MotorController import MotorController
from Lidar import Lidar

# ---------------- Control config ----------------
# TURN_SIGN = 1 means w_cmd follows x_norm; set to -1 to invert left/right turn direction
TURN_SIGN = -1
TURN_GAIN = 18   # camera-based turn responsiveness (lower = slower turning)
AVOID_TURN_FACTOR = 0.5  # scale for avoidance turns relative to TURN_GAIN (lower = gentler avoidance)
X_EMA_ALPHA = 0.4    # smoothing for x_norm
D_EMA_ALPHA = 0.35   # smoothing for lidar front distance
W_SLEW_STEP = 3      # max change per loop for w_cmd
V_SLEW_STEP = 3      # max change per loop for v_cmd
DEADZONE_W = 2       # small turn commands are zeroed
DEADZONE_V = 1       # small linear commands are zeroed
OBSTACLE_THRESH = 600  # mm threshold to consider blocked

# ---------------- Head tracking (ported from tracking_backup_19_10) ----------------
class HeadTracker:
    def __init__(self, cam_index=0, model_path="yolo11n.pt", lock_threshold=150, max_lost_frames=10):
        self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.model = YOLO(model_path)
        self.locked_target = None
        self.lock_threshold = lock_threshold
        self.target_lost_frames = 0
        self.max_lost_frames = max_lost_frames
        self._stop = False
        self.last_frame = None
        self._lock = threading.Lock()
        # Outputs
        self.found = False
        self.x_norm = 0.0  # -1..1, negative: target on left (smoothed)
        self._x_smooth = 0.0
        self.sel_head = None     # (x1,y1,x2,y2) head proxy of selected person
        self.sel_person = None   # (x1,y1,x2,y2) full person bbox of selected

    def read_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame

    @staticmethod
    def head_from_person_box(x1,y1,x2,y2):
        head_h = max(2, int((y2 - y1) * 0.25))
        return (x1, y1, x2, y1 + head_h)

    def process_once(self):
        frame = self.read_frame()
        if frame is None:
            with self._lock:
                self.found = False
                self.x_norm = 0.0
            return None
        h, w = frame.shape[:2]
        results = self.model(frame[..., ::-1], conf=0.5, classes=[0], verbose=False)
        # Collect pairs: (person_box, head_box)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                pb = (x1,y1,x2,y2)
                hb = self.head_from_person_box(x1,y1,x2,y2)
                detections.append((pb, hb))

        selected_head = None
        selected_person = None
        if detections:
            if self.locked_target is None:
                # Pick largest head area
                pb, hb = max(detections, key=lambda t:(t[1][2]-t[1][0])*(t[1][3]-t[1][1]))
                selected_head = hb
                selected_person = pb
                self.locked_target = hb
                self.target_lost_frames = 0
            else:
                lx1,ly1,lx2,ly2 = self.locked_target  # last head box
                lcx = lx1 + (lx2-lx1)//2
                lcy = ly1 + (ly2-ly1)//2
                best_hb = None; best_pb = None; best_d = 1e9
                for pb, hb in detections:
                    x1,y1,x2,y2 = hb
                    cx = x1 + (x2-x1)//2
                    cy = y1 + (y2-y1)//2
                    d = math.hypot(cx-lcx, cy-lcy)
                    if d < best_d:
                        best_d = d; best_hb = hb; best_pb = pb
                if best_hb is not None and best_d < self.lock_threshold:
                    selected_head = best_hb
                    selected_person = best_pb
                    self.locked_target = best_hb
                    self.target_lost_frames = 0
                else:
                    self.target_lost_frames += 1
                    if self.target_lost_frames >= self.max_lost_frames:
                        self.locked_target = None
                        self.target_lost_frames = 0
        # Compute x_norm
        if selected_head is not None:
            x1,y1,x2,y2 = selected_head
            cx = (x1+x2)/2.0
            x_norm = (cx / w) * 2.0 - 1.0
            # EMA smoothing for x
            self._x_smooth = (1.0 - X_EMA_ALPHA) * self._x_smooth + X_EMA_ALPHA * x_norm
            with self._lock:
                self.found = True
                self.x_norm = self._x_smooth
                self.sel_head = selected_head
                self.sel_person = selected_person
        else:
            with self._lock:
                self.found = False
                self.x_norm = 0.0
                self.sel_head = None
                self.sel_person = None
        with self._lock:
            self.last_frame = frame
        return frame

    def get_state(self):
        with self._lock:
            return self.found, self.x_norm, self.last_frame, self.sel_person, self.sel_head

    def release(self):
        self.cap.release()

# ---------------- Lidar loop for front distance ----------------
class LidarLoop:
    def __init__(self, port="COM15", baud=115200):
        self.lidar = Lidar(port, baud, has_intensity=False, model="triangle", io_timeout=0.2)
        self._stop = False
        self._lock = threading.Lock()
        self.latest_scan = None
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self._d_front_smooth = None

    def start(self):
        if not self.thread.is_alive():
            self._stop = False
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()

    def stop(self):
        self._stop = True
        if self.thread.is_alive():
            self.thread.join()
        self.lidar.close()

    def _loop(self):
        with self.lidar as ld:
            while not self._stop:
                scan = ld.get_full_rotation_dict(
                    total_timeout=0.5,
                    min_coverage_deg=180.0,
                    wrap_hysteresis_deg=40.0,
                    angle_round=1.0,
                    prefer="min",
                    max_range_mm=3000.0
                )
                with self._lock:
                    self.latest_scan = scan

    def get_front_distance(self, front_range=(175,185)):
        with self._lock:
            scan = self.latest_scan
        if scan is None:
            return None
        vals = []
        for ang in range(front_range[0], front_range[1]+1):
            d = scan.get(float(ang))
            if d is not None and d > 0:
                vals.append(d)
        if not vals:
            return None
        # Use median to be robust
        vals.sort()
        med = vals[len(vals)//2]
        # Temporal EMA smoothing
        if self._d_front_smooth is None:
            self._d_front_smooth = med
        else:
            self._d_front_smooth = (1.0 - D_EMA_ALPHA) * self._d_front_smooth + D_EMA_ALPHA * med
        return self._d_front_smooth

    def get_block_info(
        self,
        left_range=(150,179),
        right_range=(181,210),
        front_left=(170,179),
        front_right=(181,190),
        obstacle_thresh=OBSTACLE_THRESH
    ):
        """Return obstacle flags and minimal distances for sectors.
        stop condition will be evaluated in main using front_left & front_right.
        """
        with self._lock:
            scan = self.latest_scan
        info = {
            "blocked_left": False,
            "blocked_right": False,
            "front_left_blocked": False,
            "front_right_blocked": False,
            "left_min": None,
            "right_min": None,
        }
        if scan is None:
            return info
        # helper
        def sector_min(rng):
            vals = []
            for ang in range(rng[0], rng[1]+1):
                d = scan.get(float(ang))
                if d is not None and d > 0:
                    vals.append(d)
            return min(vals) if vals else None
        left_min = sector_min(left_range)
        right_min = sector_min(right_range)
        fl_min = sector_min(front_left)
        fr_min = sector_min(front_right)
        info["left_min"] = left_min
        info["right_min"] = right_min
        info["blocked_left"] = (left_min is not None and left_min < obstacle_thresh)
        info["blocked_right"] = (right_min is not None and right_min < obstacle_thresh)
        info["front_left_blocked"] = (fl_min is not None and fl_min < obstacle_thresh)
        info["front_right_blocked"] = (fr_min is not None and fr_min < obstacle_thresh)
        return info

# ---------------- Motor mixing ----------------

def mix_to_wheels(v_cmd, w_cmd, v_max=20, w_max=20):
    """Map linear v (-v_max..v_max) and angular w (-w_max..w_max) to left/right speed+dir (0..50, dir 0/1)."""
    v = max(-v_max, min(v_max, v_cmd))
    w = max(-w_max, min(w_max, w_cmd))
    left = v - w
    right = v + w
    def to_speed_dir(x):
        dir_ = 1 if x >= 0 else 0
        spd = int(min(50, max(0, abs(x))))
        return spd, dir_
    ls, ld = to_speed_dir(left)
    rs, rd = to_speed_dir(right)
    return ls, ld, rs, rd

# ---------------- Main control ----------------

def main():
    motor = MotorController(port="COM20", baudrate=115200)
    tracker = HeadTracker(cam_index=0, model_path="yolo11n.pt")
    lidar_loop = LidarLoop(port="COM15", baud=115200)
    lidar_loop.start()
    try:
        prev_v_cmd = 0
        prev_w_cmd = 0
        while True:
            frame = tracker.process_once()
            found, x_norm, last_frame, sel_person, sel_head = tracker.get_state()
            # Angular command from camera: rotate toward person
            if found:
                w_cmd_raw = int(TURN_SIGN * x_norm * TURN_GAIN)  # invert if needed
            else:
                w_cmd_raw = 0
            # Linear command from lidar: maintain 900-1000mm
            d_front = lidar_loop.get_front_distance(front_range=(175,185))
            block = lidar_loop.get_block_info(
                left_range=(140,179), right_range=(181,220),
                front_left=(170,179), front_right=(181,190),
                obstacle_thresh=OBSTACLE_THRESH
            )
            # Obstacle stop/avoid policy:
            # - Stop only if BOTH front_left and front_right are blocked
            # - If any left or right sector is blocked, turn to the free side (override camera turn)
            # Note: Apply TURN_SIGN to avoidance too so physical turn direction matches wiring.
            if block["front_left_blocked"] and block["front_right_blocked"]:
                v_cmd_raw = 0
                # hold turn 0 while stopped (or we could try to wiggle)
                w_cmd_raw = 0
            else:
                # Not a hard stop -> compute v by distance keeping
                if d_front is None or not found:
                    v_cmd_raw = 0
                else:
                    if 900 <= d_front <= 1000:
                        v_cmd_raw = 0
                    elif d_front > 1000:
                        err = min(500.0, d_front - 1000.0)
                        v_cmd_raw = int((err / 500.0) * 20)
                    else:
                        err = min(500.0, 900.0 - d_front)
                        v_cmd_raw = -int((err / 500.0) * 15)

                # Avoidance turn overrides camera when blocked on sides
                if block["blocked_left"] or block["blocked_right"]:
                    def enforce_turn_dir(current_w, desired_right, mag):
                        # desired_right=True means physically turn right; False means turn left
                        base = mag if desired_right else -mag
                        target = int(TURN_SIGN * base)
                        if target >= 0:
                            return max(current_w, target)
                        else:
                            return min(current_w, target)

                    if block["blocked_left"] and not block["blocked_right"]:
                        # obstacle on left -> turn right (respect TURN_SIGN)
                        w_cmd_raw = enforce_turn_dir(w_cmd_raw, desired_right=True, mag=int(AVOID_TURN_FACTOR * TURN_GAIN))
                    elif block["blocked_right"] and not block["blocked_left"]:
                        # obstacle on right -> turn left (respect TURN_SIGN)
                        w_cmd_raw = enforce_turn_dir(w_cmd_raw, desired_right=False, mag=int(AVOID_TURN_FACTOR * TURN_GAIN))
                    else:
                        # both sides flagged; turn toward larger clearance (respect TURN_SIGN)
                        lmin = block["left_min"] if block["left_min"] is not None else 0
                        rmin = block["right_min"] if block["right_min"] is not None else 0
                        if lmin < rmin:
                            w_cmd_raw = enforce_turn_dir(w_cmd_raw, desired_right=True, mag=int(AVOID_TURN_FACTOR * TURN_GAIN))
                        else:
                            w_cmd_raw = enforce_turn_dir(w_cmd_raw, desired_right=False, mag=int(AVOID_TURN_FACTOR * TURN_GAIN))
                            
                        #Turn back 
                        
            if d_front is None or not found:
                v_cmd_raw = 0
            else:
                # Deadband 900..1000
                if 900 <= d_front <= 1000:
                    v_cmd_raw = 0
                elif d_front > 1000:
                    # forward proportional
                    err = min(500.0, d_front - 1000.0)  # cap error
                    v_cmd_raw = int((err / 500.0) * 20)  # up to 20
                else:
                    # too close: move backward
                    err = min(400.0, 900.0 - d_front)
                    v_cmd_raw = -int((err / 400.0) * 20)  # backward slower up to -20

            # Apply deadzones
            if abs(w_cmd_raw) < DEADZONE_W:
                w_cmd_raw = 0
            if abs(v_cmd_raw) < DEADZONE_V:
                v_cmd_raw = 0

            # Slew-rate limiters
            def slew(prev, target, step):
                if target > prev + step:
                    return prev + step
                if target < prev - step:
                    return prev - step
                return target

            w_cmd = slew(prev_w_cmd, w_cmd_raw, W_SLEW_STEP)
            v_cmd = slew(prev_v_cmd, v_cmd_raw, V_SLEW_STEP)
            prev_w_cmd = w_cmd
            prev_v_cmd = v_cmd

            ls, ld, rs, rd = mix_to_wheels(v_cmd, w_cmd, v_max=20, w_max=12)
            motor.send_if_changed(ls, ld, rs, rd)
            # Optional view
            if last_frame is not None:
                h, w = last_frame.shape[:2]
                cx = int((x_norm*0.5 + 0.5) * w)
                cv2.line(last_frame, (w//2, 0), (w//2, h), (0,255,255), 1)
                cv2.line(last_frame, (cx, 0), (cx, h), (0,0,255), 1)
                # Draw person bbox and head bbox for clearer visualization
                if sel_person is not None:
                    px1,py1,px2,py2 = map(int, sel_person)
                    cv2.rectangle(last_frame, (px1,py1), (px2,py2), (0,200,255), 2)
                    cv2.putText(last_frame, "PERSON", (px1, max(0,py1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)
                if sel_head is not None:
                    hx1,hy1,hx2,hy2 = map(int, sel_head)
                    cv2.rectangle(last_frame, (hx1,hy1), (hx2,hy2), (0,255,0), 2)
                    cv2.putText(last_frame, "HEAD", (hx1, max(0,hy1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                txt = f"found={found} x={x_norm:.2f} d_front={d_front if d_front is not None else 'NA'} v={v_cmd} w={w_cmd}"
                cv2.putText(last_frame, txt, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                cv2.imshow("Head+Lidar Control", last_frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    # ESC pressed -> immediately stop robot
                    motor.send_if_changed(0, 1, 0, 1)
                    break
            time.sleep(0.05)
    finally:
        # Ensure robot is stopped on exit
        try:
            motor.send_if_changed(0, 1, 0, 1)
        except Exception:
            pass
        lidar_loop.stop()
        tracker.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
