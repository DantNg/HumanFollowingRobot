import cv2
from ultralytics import YOLO
import serial

# Path to YOLO person detection model (for head tracking)
MODEL_PATH = "yolo11n.pt"  # Person detection model

# Load YOLO model
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(0)
# --- Serial setup ---
SERIAL_PORT = "COM19"  # Đổi thành cổng của bạn
SERIAL_BAUD = 115200
try:
    ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.1)
    print(f"[SERIAL] Connected to {SERIAL_PORT}")
except Exception as e:
    print(f"[SERIAL] Error: {e}")
    ser = None

def send_motor_command(left_speed, left_dir, right_speed, right_dir):
    """
    Gửi lệnh điều khiển động cơ dạng:
    L<speed>-<dir>\r\n
    R<speed>-<dir>\r\n
    left_speed, right_speed: 0-50
    left_dir, right_dir: 1 (forward), 0 (backward)
    """
    if ser and ser.is_open:
        l_cmd = f"L{left_speed}-{left_dir}\r\n"
        r_cmd = f"R{right_speed}-{right_dir}\r\n"
        ser.write(l_cmd.encode())
        ser.write(r_cmd.encode())
        print(f"[SEND] {l_cmd.strip()} | {r_cmd.strip()}")
    else:
        print(f"[SIM] L{left_speed}-{left_dir} | R{right_speed}-{right_dir}")

# --- Thêm biến lưu lệnh trước đó ---
last_command = None

# --- Thêm biến cho target locking ---
locked_target = None  # Lưu bbox của mục tiêu đã khóa (x1, y1, x2, y2)
lock_threshold = 150  # Khoảng cách pixel tối đa để xác định cùng mục tiêu
target_lost_frames = 0  # Đếm số frame mất mục tiêu
max_lost_frames = 10  # Số frame tối đa cho phép mất mục tiêu trước khi reset

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO expects RGB - detect persons
    results = model(frame[..., ::-1], conf=0.5, classes=[0])  # Only detect person (class 0)
    h, w = frame.shape[:2]

    found_person = False  # <--- Thêm biến này
    selected_box = None  # Bbox được chọn để điều khiển

    # --- Target Locking Algorithm for Head Tracking ---
    if results:
        all_boxes = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Extract head region (top 1/4 of person bbox)
                head_height = (y2 - y1) // 4
                head_x1, head_y1 = x1, y1
                head_x2, head_y2 = x2, y1 + head_height
                all_boxes.append((head_x1, head_y1, head_x2, head_y2))

        if all_boxes:
            if locked_target is None:
                # Chưa có mục tiêu, chọn bbox lớn nhất (gần nhất)
                largest_area = 0
                for box in all_boxes:
                    x1, y1, x2, y2 = box
                    area = (x2 - x1) * (y2 - y1)
                    if area > largest_area:
                        largest_area = area
                        selected_box = box
                        locked_target = box
                print(f"[LOCK] New head target locked at {locked_target}")
            else:
                # Đã có mục tiêu, tìm bbox gần nhất với mục tiêu đã khóa
                lx1, ly1, lx2, ly2 = locked_target
                lcx = lx1 + (lx2 - lx1) // 2
                lcy = ly1 + (ly2 - ly1) // 2
                
                min_distance = float('inf')
                closest_box = None
                
                for box in all_boxes:
                    x1, y1, x2, y2 = box
                    cx = x1 + (x2 - x1) // 2
                    cy = y1 + (y2 - y1) // 2
                    distance = ((cx - lcx)**2 + (cy - lcy)**2)**0.5
                    
                    if distance < min_distance:
                        min_distance = distance
                        closest_box = box
                
                # Nếu tìm thấy bbox gần với mục tiêu đã khóa
                if closest_box and min_distance < lock_threshold:
                    selected_box = closest_box
                    locked_target = closest_box  # Cập nhật vị trí mục tiêu
                    target_lost_frames = 0
                    found_person = True
                    print(f"[TRACK] Following head target, distance: {min_distance:.1f}")
                else:
                    # Mất mục tiêu
                    target_lost_frames += 1
                    print(f"[LOST] Head target lost for {target_lost_frames} frames")
                    
                    if target_lost_frames >= max_lost_frames:
                        # Reset target lock sau khi mất quá lâu
                        locked_target = None
                        target_lost_frames = 0
                        print("[RESET] Head target lock reset")

    # Xử lý điều khiển robot chỉ khi có mục tiêu được chọn
    if selected_box:
        found_person = True
        x1, y1, x2, y2 = selected_box
    # Xử lý điều khiển robot chỉ khi có mục tiêu được chọn
    if selected_box:
        found_person = True
        x1, y1, x2, y2 = selected_box
        fw, fh = x2 - x1, y2 - y1
        cx = x1 + fw // 2
        cy = y1 + fh // 2

        # Left/Right position
        if cx < w // 3:
            pos_lr = "Left"
        elif cx > 2 * w // 3:
            pos_lr = "Right"
        else:
            pos_lr = "Center"

        # Far/Near based on bbox height
        ratio = fh / h
        if ratio > 0.3:
            pos_fb = "Very Close"
        elif ratio > 0.15:
            pos_fb = "Close"
        elif ratio > 0.07:
            pos_fb = "Medium"
        else:
            pos_fb = "Far"

        # --- Điều khiển robot ---
        # Nếu Center và Close thì dừng lại
        if pos_lr == "Center" and pos_fb == "Close":
            left_speed = right_speed = 0
            left_dir = right_dir = 1
        else:
            if pos_fb == "Very Close":
                speed = 0
            elif pos_fb == "Close":
                speed = 15
            elif pos_fb == "Medium":
                speed = 20
            else:
                speed = 50

            if pos_lr == "Center":
                left_speed = right_speed = speed
                left_dir = right_dir = 1
            elif pos_lr == "Left":
                left_speed = int(speed * 0.5)
                right_speed = speed
                left_dir = right_dir = 1
            elif pos_lr == "Right":
                left_speed = speed
                right_speed = int(speed * 0.5)
                left_dir = right_dir = 1
            else:
                left_speed = right_speed = 0
                left_dir = right_dir = 1

        # --- Chỉ gửi lệnh khi thay đổi ---
        command_tuple = (left_speed, left_dir, right_speed, right_dir)
        if command_tuple != last_command:
            send_motor_command(left_speed, left_dir, right_speed, right_dir)
            last_command = command_tuple

        # Vẽ bbox và thông tin cho head tracking
        label = f"{pos_lr} - {pos_fb} [HEAD LOCKED]"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 3)  # Bbox xanh cho head region
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        
        # Hiển thị tốc độ 2 bánh
        speed_info = f"L:{left_speed} R:{right_speed}"
        cv2.putText(frame, speed_info, (x1, y2+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,100,100), 2)
        
        # Vẽ target lock indicator cho head
        cv2.circle(frame, (cx, cy), 8, (0, 0, 255), 2)  # Điểm đỏ ở trung tâm đầu
        cv2.putText(frame, "HEAD", (cx-20, cy-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Nếu không phát hiện người, dừng robot ngay lập tức
    if not found_person:
        stop_command = (0, 1, 0, 1)
        if last_command != stop_command:
            send_motor_command(0, 1, 0, 1)
            last_command = stop_command

    cv2.imshow("YOLO Head Tracking", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()