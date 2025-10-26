import cv2
import numpy as np
from ultralytics import YOLO
import serial
import math

# ================== Config ==================
MODEL_PATH = "yolo11n.pt"   # Person detection model
CAM_INDEX  = 1

SERIAL_PORT = "COM19"
SERIAL_BAUD = 115200

CONF_PERSON = 0.5
HEAD_STRIPE = (0.0, 0.35)
HEAD_IOU_REINIT = 0.25
TRACKER_HOLD_FRAMES = 12

CENTER_ZONE = 1/3
FAR_THR   = 0.07
MED_THR   = 0.15
CLOSE_THR = 0.30
MAX_PWM = 50

# ================== Model & Serial ==================
model = YOLO(MODEL_PATH)

try:
    ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.1)
    print(f"[SERIAL] Connected to {SERIAL_PORT}")
except Exception as e:
    print(f"[SERIAL] Error: {e}")
    ser = None

def send_motor_command(left_speed, left_dir, right_speed, right_dir):
    if ser and ser.is_open:
        l_cmd = f"L{left_speed}-{left_dir}\r\n"
        r_cmd = f"R{right_speed}-{right_dir}\r\n"
        ser.write(l_cmd.encode())
        ser.write(r_cmd.encode())
        print(f"[SEND] {l_cmd.strip()} | {r_cmd.strip()}")
    else:
        print(f"[SIM] L{left_speed}-{left_dir} | R{right_speed}-{right_dir}")

last_command = None
def send_if_changed(L, Ld, R, Rd):
    global last_command
    tup = (L,Ld,R,Rd)
    if tup != last_command:
        send_motor_command(L,Ld,R,Rd)
        last_command = tup

# ================== Tracker Compatibility ==================
def create_tracker_anywhere():
    """
    Trả về tracker khả dụng (CSRT > KCF > MOSSE > MIL).
    Tự động tương thích mọi phiên bản OpenCV.
    """
    tracker_constructors = [
        # Các tên khả dụng tùy phiên bản
        "TrackerCSRT_create",
        "legacy.TrackerCSRT_create",
        "TrackerKCF_create",
        "legacy.TrackerKCF_create",
        "TrackerMOSSE_create",
        "legacy.TrackerMOSSE_create",
        "TrackerMIL_create",
        "legacy.TrackerMIL_create",
    ]
    for name in tracker_constructors:
        parts = name.split(".")
        mod = cv2
        ok = True
        for p in parts:
            if not hasattr(mod, p):
                ok = False
                break
            mod = getattr(mod, p)
        if ok and callable(mod):
            print(f"[TRACKER] Using {name}")
            return mod()
    raise RuntimeError("No compatible tracker found. Please install opencv-contrib-python")

# ================== Helper funcs ==================
def iou(a,b):
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    inter = max(0,min(ax2,bx2)-max(ax1,bx1)) * max(0,min(ay2,by2)-max(ay1,by1))
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter/ua if ua>0 else 0.0

def to_xywh(box):
    x1,y1,x2,y2 = box
    return (int(x1), int(y1), int(x2-x1), int(y2-y1))

def make_head_box_from_person(person_box):
    if person_box is None: return None
    x1,y1,x2,y2 = person_box
    h = y2 - y1
    top = int(y1)
    bot = int(y1 + HEAD_STRIPE[1]*h)
    if bot <= top: bot = top + 2
    return (x1, top, x2, bot)

# ================== Main ==================
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(3, 1280)
cap.set(4, 720)

head_tracker = None
use_head_tracker = False
tracker_grace = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    selected_box = None
    found_person = False

    results = model(frame[..., ::-1], conf=CONF_PERSON, classes=[0], verbose=False)
    persons = []
    for r in results:
        for box in r.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            area = (x2-x1)*(y2-y1)
            persons.append(((x1,y1,x2,y2), area))
    if persons:
        persons.sort(key=lambda it: it[1], reverse=True)
        selected_box = persons[0][0]
        found_person = True
        head_proxy = make_head_box_from_person(selected_box)
        if not use_head_tracker or head_tracker is None:
            head_tracker = create_tracker_anywhere()
            head_tracker.init(frame, to_xywh(head_proxy))
            use_head_tracker = True
            tracker_grace = TRACKER_HOLD_FRAMES
        else:
            ok, box_xywh = head_tracker.update(frame)
            if ok:
                x, y, w_, h_ = map(int, box_xywh)
                head_now = (x, y, x+w_, y+h_)
                if iou(head_now, head_proxy) < HEAD_IOU_REINIT:
                    head_tracker = create_tracker_anywhere()
                    head_tracker.init(frame, to_xywh(head_proxy))
            else:
                head_tracker = create_tracker_anywhere()
                head_tracker.init(frame, to_xywh(head_proxy))
    else:
        if use_head_tracker and head_tracker is not None and tracker_grace > 0:
            found_person = True
            tracker_grace -= 1
        else:
            use_head_tracker = False
            head_tracker = None
            tracker_grace = 0

    head_box_to_use = None
    if use_head_tracker and head_tracker is not None:
        ok, box_xywh = head_tracker.update(frame)
        if ok:
            x, y, w_, h_ = map(int, box_xywh)
            head_box_to_use = (x, y, x+w_, y+h_)

    if head_box_to_use is None and selected_box is not None:
        head_box_to_use = make_head_box_from_person(selected_box)

    # ----- Điều khiển robot -----
    if head_box_to_use is not None:
        x1,y1,x2,y2 = head_box_to_use
        fw, fh = x2-x1, y2-y1
        cx = x1 + fw//2
        if cx < w*(1/3): pos_lr = "Left"
        elif cx > w*(2/3): pos_lr = "Right"
        else: pos_lr = "Center"

        src = selected_box if selected_box else head_box_to_use
        rx1, ry1, rx2, ry2 = src
        ratio = (ry2-ry1)/h

        if ratio > CLOSE_THR: pos_fb, speed = "Very Close", 0
        elif ratio > MED_THR: pos_fb, speed = "Close", 15
        elif ratio > FAR_THR: pos_fb, speed = "Medium", 20
        else: pos_fb, speed = "Far", 50

        if pos_lr == "Center":
            L = R = speed
        elif pos_lr == "Left":
            L = int(speed*0.5); R = speed
        elif pos_lr == "Right":
            L = speed; R = int(speed*0.5)
        else:
            L = R = 0

        send_if_changed(L,1,R,1)

        # Vẽ head + person
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 3)
        cv2.putText(frame, f"{pos_lr} - {pos_fb} [HEAD]", (x1,max(0,y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,255),2)
    else:
        send_if_changed(0,1,0,1)
        cv2.putText(frame,"LOST", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0,0,255),2)

    cv2.imshow("Follow Head/Person", frame)
    if cv2.waitKey(1)&0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()
