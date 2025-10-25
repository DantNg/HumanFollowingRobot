import cv2
from ultralytics import YOLO
import serial

# Path to YOLO face detection model (change to yolov11n-face.pt if you have it)
MODEL_PATH = "yolov11n-face.pt"

# Load YOLO model
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(0)

# --- Serial setup ---
SERIAL_PORT = "COM16"  # Đổi thành cổng của bạn
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

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO expects RGB
    results = model(frame[..., ::-1], conf=0.5)
    h, w = frame.shape[:2]

    found_person = False  # <--- Thêm biến này

    for r in results:
        for box in r.boxes:
            found_person = True  # <--- Đánh dấu có người
            x1, y1, x2, y2 = map(int, box.xyxy[0])
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

            label = f"{pos_lr} - {pos_fb}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            # Hiển thị tốc độ 2 bánh
            speed_info = f"L:{left_speed} R:{right_speed}"
            cv2.putText(frame, speed_info, (x1, y2+25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,100,100), 2)

    # Nếu không phát hiện người, dừng robot ngay lập tức
    if not found_person:
        stop_command = (0, 1, 0, 1)
        if last_command != stop_command:
            send_motor_command(0, 1, 0, 1)
            last_command = stop_command

    cv2.imshow("YOLO Face Position", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()