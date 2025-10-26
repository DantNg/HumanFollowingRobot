import cv2, numpy as np, time, math
from ultralytics import YOLO
import serial

# ================== Config ==================
FACE_WEIGHTS   = "yolov11n-face.pt"   # model face
PERSON_WEIGHTS = "yolo11n.pt"         # model person (class 0)
CAM_INDEX      = 1
CONF_FACE      = 0.4
CONF_PERSON    = 0.5

SERIAL_PORT = "COM16"
SERIAL_BAUD = 115200

# Điều khiển
MAX_PWM = 20           # 0..50 theo MCU của bạn
CENTER_ZONE = 0.15     # |x_norm| < 0.15 coi là Center
FAR_THR   = 0.07       # các ngưỡng ratio dùng cho bucket speed
MED_THR   = 0.15
CLOSE_THR = 0.30
KX = 1.4               # gain quay
SMOOTH = 0.5           # mượt hóa x_norm

# Lock / track
LOCK_DIST_PX = 160     # khoảng cách tối đa coi là cùng mục tiêu (trên ảnh)
MAX_LOST_FACE_FRAMES   = 8
MAX_LOST_PERSON_FRAMES = 20

# -------------- Serial --------------
try:
    ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.1)
    print(f"[SERIAL] Connected {SERIAL_PORT}")
except Exception as e:
    print(f"[SERIAL] Error: {e}")
    ser = None

def send_motor_command(left_speed, left_dir, right_speed, right_dir):
    if ser and ser.is_open:
        l_cmd = f"L{left_speed}-{left_dir}\r\n"
        r_cmd = f"R{right_speed}-{right_dir}\r\n"
        ser.write(l_cmd.encode()); ser.write(r_cmd.encode())
    else:
        print(f"[SIM] L{left_speed}-{left_dir} | R{right_speed}-{right_dir}")

last_cmd = None
def send_if_changed(L, Ld, R, Rd):
    global last_cmd
    tup = (L,Ld,R,Rd)
    if tup != last_cmd:
        send_motor_command(L,Ld,R,Rd)
        last_cmd = tup

# -------------- Models --------------
face_model   = YOLO(FACE_WEIGHTS)
person_model = YOLO(PERSON_WEIGHTS)

# -------------- Helpers --------------
def cxcywh(x1,y1,x2,y2):
    w = x2-x1; h = y2-y1
    return (x1+w*0.5, y1+h*0.5, w, h)

def box_center(box):
    x1,y1,x2,y2 = box
    return ((x1+x2)/2.0, (y1+y2)/2.0)

def iou(a,b):
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    inter = max(0,min(ax2,bx2)-max(ax1,bx1))*max(0,min(ay2,by2)-max(ay1,by1))
    ua = (ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
    return inter/ua if ua>0 else 0.0

def l2(p,q): return math.hypot(p[0]-q[0], p[1]-q[1])

def color_hist(img, box):
    x1,y1,x2,y2 = [int(v) for v in box]
    x1=max(0,x1); y1=max(0,y1); x2=min(img.shape[1]-1,x2); y2=min(img.shape[0]-1,y2)
    if x2<=x1 or y2<=y1: return None
    hsv = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv],[0],None,[16],[0,180])
    h = cv2.normalize(h, None).flatten()
    return h

def hist_cos(a,b):
    if a is None or b is None: return 0.0
    na = np.linalg.norm(a); nb=np.linalg.norm(b)
    if na<1e-6 or nb<1e-6: return 0.0
    return float(np.dot(a,b)/(na*nb))

# -------------- Target state --------------
state = {
    "face_lock": None,         # bbox face hiện tại
    "person_lock": None,       # bbox person hiện tại
    "face_lost": 0,
    "person_lost": 0,
    "shirt_hist": None,        # histogram vùng thân để re-ID nhẹ
    "x_smooth": 0.0
}

# -------------- Distance estimation --------------
# Bạn có thể calibrate: dist ≈ k / face_height_px (đo ở d_ref để tìm k)
def estimate_dist_from_face(face_h_px, img_h, k_ref=0.6):
    # k_ref ~ H_face_norm * dist_ref (thô), bạn nên đo thực tế để thay!
    face_ratio = face_h_px / max(1.0, img_h)
    if face_ratio < 1e-3: return None
    # ví dụ: ở ~1.5m, face_ratio ~ 0.12 => k ≈ 0.18 (tuỳ camera)
    # tạm thời tuyến tính nghịch đảo:
    return k_ref / face_ratio

def estimate_dist_from_person(person_h_px, img_h, k_ref=1.9):
    ratio = person_h_px / max(1.0, img_h)
    if ratio < 1e-3: return None
    return k_ref / ratio

# -------------- Control from image-only --------------
def speed_bucket_from_ratio(ratio):
    if ratio > CLOSE_THR:      sp = 0
    elif ratio > MED_THR:      sp = 15
    elif ratio > FAR_THR:      sp = 20
    else:                      sp = 50
    return sp

# ============== Main loop ==============
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ok, frame = cap.read()
    if not ok: break
    H,W = frame.shape[:2]

    # --- 1) Detect face & person ---
    faces = []
    res_f = face_model(frame[...,::-1], conf=CONF_FACE, verbose=False)[0]
    for b in res_f.boxes:
        x1,y1,x2,y2 = map(int, b.xyxy[0])
        faces.append((x1,y1,x2,y2))

    persons = []
    res_p = person_model(frame[...,::-1], conf=CONF_PERSON, classes=[0], verbose=False)[0]
    for b in res_p.boxes:
        x1,y1,x2,y2 = map(int, b.xyxy[0])
        persons.append((x1,y1,x2,y2))

    # --- 2) Update face lock ---
    sel_face = None
    if faces:
        if state["face_lock"] is None:
            # chọn mặt có diện tích lớn nhất (gần nhất) để lock
            sel_face = max(faces, key=lambda bb:(bb[2]-bb[0])*(bb[3]-bb[1]))
            state["face_lock"] = sel_face
            state["face_lost"] = 0
        else:
            # match theo gần tâm
            fx,fy = box_center(state["face_lock"])
            dmin, best = 1e9, None
            for bb in faces:
                cx,cy = box_center(bb)
                d = l2((fx,fy),(cx,cy))
                if d < dmin:
                    dmin, best = d, bb
            if dmin < LOCK_DIST_PX:
                sel_face = best
                state["face_lock"] = sel_face
                state["face_lost"] = 0
            else:
                state["face_lost"] += 1
    else:
        state["face_lost"] += 1

    if state["face_lost"] > MAX_LOST_FACE_FRAMES:
        state["face_lock"] = None
        state["face_lost"] = 0

    # --- 3) Attach/gán person theo face khi có mặt ---
    sel_person = None
    if persons:
        if state["face_lock"] is not None:
            # chọn person chứa mặt hoặc IOU/center gần mặt
            fx1,fy1,fx2,fy2 = state["face_lock"]
            fcx,fcy = box_center(state["face_lock"])
            best_score, best_bb = -1, None
            for pb in persons:
                px1,py1,px2,py2 = pb
                contains = (fx1>=px1 and fy1>=py1 and fx2<=px2 and fy2<=py2)
                score = 0.0
                if contains: score += 2.0
                score += iou(pb, state["face_lock"])  # ưu tiên khớp vị trí
                pcx,pcy = box_center(pb)
                score += max(0.0, 1.0 - l2((fcx,fcy),(pcx,pcy))/LOCK_DIST_PX)
                if score > best_score:
                    best_score, best_bb = score, pb
            sel_person = best_bb
            state["person_lock"] = sel_person
            state["person_lost"] = 0
            # cập nhật màu áo (vùng 40–70% chiều cao bbox)
            if sel_person is not None:
                x1,y1,x2,y2 = sel_person
                hh = y2-y1
                upper = (x1, y1+int(0.4*hh), x2, y1+int(0.7*hh))
                state["shirt_hist"] = color_hist(frame, upper)
        else:
            # không có mặt → duy trì person lock cũ bằng IOU + màu áo
            if state["person_lock"] is not None:
                best = None; best_s = -1
                for pb in persons:
                    s = 0.0
                    s += 1.5*iou(pb, state["person_lock"])
                    # màu áo
                    h0 = state["shirt_hist"]
                    x1,y1,x2,y2 = pb; hh=y2-y1
                    upper=(x1, y1+int(0.4*hh), x2, y1+int(0.7*hh))
                    h1 = color_hist(frame, upper)
                    s += 0.8*hist_cos(h0,h1)
                    # gần tâm cũ
                    s += max(0.0, 1.0 - l2(box_center(pb), box_center(state["person_lock"]))/(LOCK_DIST_PX*1.2))
                    if s > best_s: best_s, best = s, pb
                if best is not None:
                    sel_person = best
                    state["person_lock"] = sel_person
                    state["person_lost"] = 0
                else:
                    state["person_lost"] += 1
            else:
                # chưa có lock: chọn person lớn nhất
                sel_person = max(persons, key=lambda bb:(bb[2]-bb[0])*(bb[3]-bb[1]))
                state["person_lock"] = sel_person
                state["person_lost"] = 0
    else:
        state["person_lost"] += 1

    if state["person_lost"] > MAX_LOST_PERSON_FRAMES:
        state["person_lock"] = None
        state["person_lost"] = 0
        state["shirt_hist"] = None

    # --- 4) Quyết định điều khiển từ box (ưu tiên face để tính khoảng cách) ---
    found = False
    x_norm = 0.0
    speed_pct = 0
    label_ex = ""

    # chọn box điều hướng (center dùng để quay): ưu tiên person_lock (ổn định hơn)
    nav_box = state["person_lock"] if state["person_lock"] is not None else state["face_lock"]
    if nav_box is not None:
        found = True
        x1,y1,x2,y2 = nav_box
        cx = (x1+x2)*0.5; cy=(y1+y2)*0.5
        x_norm_raw = (cx / W)*2.0 - 1.0
        state["x_smooth"] = (1-SMOOTH)*state["x_smooth"] + SMOOTH*x_norm_raw
        x_norm = state["x_smooth"]

        # tính “khoảng cách” từ face nếu có, nếu không dùng person
        dist_est = None
        if state["face_lock"] is not None:
            fx1,fy1,fx2,fy2 = state["face_lock"]
            dist_est = estimate_dist_from_face(fy2-fy1, H, k_ref=0.18)  # gợi ý k_ref ~0.18–0.22 (cần calibrate)
            label_ex = "FACE-LOCK"
        else:
            dist_est = estimate_dist_from_person(y2-y1, H, k_ref=1.9)    # gợi ý k_ref ~1.7–2.2
            label_ex = "PERSON-LOCK"

        # bucket tốc độ từ tỷ lệ bbox (đơn giản, bạn có thể thay bằng dist_est)
        ratio = (y2-y1)/H
        speed_pct = speed_bucket_from_ratio(ratio)

        # mapping quay theo x_norm
        # Nếu gần center, chạy thẳng; lệch thì giảm 1 bên
        if abs(x_norm) < CENTER_ZONE:
            L = R = speed_pct
        elif x_norm < 0:
            # mục tiêu bên trái → giảm bánh trái
            L = int(speed_pct*0.5); R = speed_pct
        else:
            L = speed_pct; R = int(speed_pct*0.5)

        send_if_changed(L,1,R,1)

        # vẽ
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(frame, f"{label_ex} x={x_norm:.2f} sp={speed_pct}", (x1,max(0,y1-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv2.circle(frame,(int(cx),int(cy)),6,(0,0,255),2)

    if not found:
        # dừng an toàn
        send_if_changed(0,1,0,1)
        cv2.putText(frame, "LOST: searching...", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    # vẽ face lock (nếu khác nav_box)
    if state["face_lock"] is not None and nav_box is not state["face_lock"]:
        x1,y1,x2,y2 = state["face_lock"]
        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,200,0),2)
        cv2.putText(frame, "FACE", (x1,max(0,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,200,0), 2)

    # vẽ person lock (nếu khác nav_box)
    if state["person_lock"] is not None and nav_box is not state["person_lock"]:
        x1,y1,x2,y2 = state["person_lock"]
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,200,255),2)
        cv2.putText(frame, "PERSON", (x1,max(0,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,255), 2)

    cv2.imshow("Face->Person ID Lock & Handoff", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
