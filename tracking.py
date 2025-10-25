
import cv2, numpy as np, time, math
from ultralytics import YOLO

class HumanTracking:
    def get_action_from_state(self, x_norm, speed_pct, nav_box):
        """Trả về action: 'forward', 'left', 'right', 'stop' dựa vào trạng thái tracking."""
        if nav_box is None:
            return "stop"
        if abs(x_norm) < self.CENTER_ZONE:
            if speed_pct > 0:
                return "forward"
            else:
                return "stop"
        elif x_norm < 0:
            return "left"
        else:
            return "right"
    FACE_WEIGHTS   = "yolov11n-face.pt"
    PERSON_WEIGHTS = "yolo11n.pt"
    CAM_INDEX      = 0
    CONF_FACE      = 0.4
    CONF_PERSON    = 0.5
    MAX_PWM = 50
    CENTER_ZONE = 0.15
    FAR_THR   = 0.07
    MED_THR   = 0.15
    CLOSE_THR = 0.30
    KX = 1.4
    SMOOTH = 0.5
    LOCK_DIST_PX = 160
    MAX_LOST_FACE_FRAMES   = 8
    MAX_LOST_PERSON_FRAMES = 20

    def __init__(self):
        self.face_model   = YOLO(self.FACE_WEIGHTS)
        self.person_model = YOLO(self.PERSON_WEIGHTS)
        self.state = {
            "face_lock": None,
            "person_lock": None,
            "face_lost": 0,
            "person_lost": 0,
            "shirt_hist": None,
            "x_smooth": 0.0
        }

    @staticmethod
    def cxcywh(x1,y1,x2,y2):
        w = x2-x1; h = y2-y1
        return (x1+w*0.5, y1+h*0.5, w, h)

    @staticmethod
    def box_center(box):
        x1,y1,x2,y2 = box
        return ((x1+x2)/2.0, (y1+y2)/2.0)

    @staticmethod
    def iou(a,b):
        ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
        inter = max(0,min(ax2,bx2)-max(ax1,bx1))*max(0,min(ay2,by2)-max(ay1,by1))
        ua = (ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
        return inter/ua if ua>0 else 0.0

    @staticmethod
    def l2(p,q): return math.hypot(p[0]-q[0], p[1]-q[1])

    @staticmethod
    def color_hist(img, box):
        x1,y1,x2,y2 = [int(v) for v in box]
        x1=max(0,x1); y1=max(0,y1); x2=min(img.shape[1]-1,x2); y2=min(img.shape[0]-1,y2)
        if x2<=x1 or y2<=y1: return None
        hsv = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv],[0],None,[16],[0,180])
        h = cv2.normalize(h, None).flatten()
        return h

    @staticmethod
    def hist_cos(a,b):
        if a is None or b is None: return 0.0
        na = np.linalg.norm(a); nb=np.linalg.norm(b)
        if na<1e-6 or nb<1e-6: return 0.0
        return float(np.dot(a,b)/(na*nb))

    @staticmethod
    def estimate_dist_from_face(face_h_px, img_h, k_ref=0.6):
        face_ratio = face_h_px / max(1.0, img_h)
        if face_ratio < 1e-3: return None
        return k_ref / face_ratio

    @staticmethod
    def estimate_dist_from_person(person_h_px, img_h, k_ref=1.9):
        ratio = person_h_px / max(1.0, img_h)
        if ratio < 1e-3: return None
        return k_ref / ratio

    def speed_bucket_from_ratio(self, ratio):
        if ratio > self.CLOSE_THR:      sp = 0
        elif ratio > self.MED_THR:      sp = 15
        elif ratio > self.FAR_THR:      sp = 20
        else:                          sp = 50
        return sp

    def process_frame(self, frame):
        state = self.state
        H,W = frame.shape[:2]
        faces = []
        res_f = self.face_model(frame[...,::-1], conf=self.CONF_FACE, verbose=False)[0]
        for b in res_f.boxes:
            x1,y1,x2,y2 = map(int, b.xyxy[0])
            faces.append((x1,y1,x2,y2))
        persons = []
        res_p = self.person_model(frame[...,::-1], conf=self.CONF_PERSON, classes=[0], verbose=False)[0]
        for b in res_p.boxes:
            x1,y1,x2,y2 = map(int, b.xyxy[0])
            persons.append((x1,y1,x2,y2))

        # --- 2) Update face lock ---
        sel_face = None
        if faces:
            if state["face_lock"] is None:
                sel_face = max(faces, key=lambda bb:(bb[2]-bb[0])*(bb[3]-bb[1]))
                state["face_lock"] = sel_face
                state["face_lost"] = 0
            else:
                fx,fy = self.box_center(state["face_lock"])
                dmin, best = 1e9, None
                for bb in faces:
                    cx,cy = self.box_center(bb)
                    d = self.l2((fx,fy),(cx,cy))
                    if d < dmin:
                        dmin, best = d, bb
                if dmin < self.LOCK_DIST_PX:
                    sel_face = best
                    state["face_lock"] = sel_face
                    state["face_lost"] = 0
                else:
                    state["face_lost"] += 1
        else:
            state["face_lost"] += 1

        if state["face_lost"] > self.MAX_LOST_FACE_FRAMES:
            state["face_lock"] = None
            state["face_lost"] = 0

        # --- 3) Attach/gán person theo face khi có mặt ---
        sel_person = None
        if persons:
            if state["face_lock"] is not None:
                fx1,fy1,fx2,fy2 = state["face_lock"]
                fcx,fcy = self.box_center(state["face_lock"])
                best_score, best_bb = -1, None
                for pb in persons:
                    px1,py1,px2,py2 = pb
                    contains = (fx1>=px1 and fy1>=py1 and fx2<=px2 and fy2<=py2)
                    score = 0.0
                    if contains: score += 2.0
                    score += self.iou(pb, state["face_lock"])
                    pcx,pcy = self.box_center(pb)
                    score += max(0.0, 1.0 - self.l2((fcx,fcy),(pcx,pcy))/self.LOCK_DIST_PX)
                    if score > best_score:
                        best_score, best_bb = score, pb
                sel_person = best_bb
                state["person_lock"] = sel_person
                state["person_lost"] = 0
                if sel_person is not None:
                    x1,y1,x2,y2 = sel_person
                    hh = y2-y1
                    upper = (x1, y1+int(0.4*hh), x2, y1+int(0.7*hh))
                    state["shirt_hist"] = self.color_hist(frame, upper)
            else:
                if state["person_lock"] is not None:
                    best = None; best_s = -1
                    for pb in persons:
                        s = 0.0
                        s += 1.5*self.iou(pb, state["person_lock"])
                        h0 = state["shirt_hist"]
                        x1,y1,x2,y2 = pb; hh=y2-y1
                        upper=(x1, y1+int(0.4*hh), x2, y1+int(0.7*hh))
                        h1 = self.color_hist(frame, upper)
                        s += 0.8*self.hist_cos(h0,h1)
                        s += max(0.0, 1.0 - self.l2(self.box_center(pb), self.box_center(state["person_lock"]))/(self.LOCK_DIST_PX*1.2))
                        if s > best_s: best_s, best = s, pb
                    if best is not None:
                        sel_person = best
                        state["person_lock"] = sel_person
                        state["person_lost"] = 0
                    else:
                        state["person_lost"] += 1
                else:
                    sel_person = max(persons, key=lambda bb:(bb[2]-bb[0])*(bb[3]-bb[1]))
                    state["person_lock"] = sel_person
                    state["person_lost"] = 0
        else:
            state["person_lost"] += 1

        if state["person_lost"] > self.MAX_LOST_PERSON_FRAMES:
            state["person_lock"] = None
            state["person_lost"] = 0
            state["shirt_hist"] = None

        # --- 4) Quyết định điều khiển từ box (ưu tiên face để tính khoảng cách) ---
        found = False
        x_norm = 0.0
        speed_pct = 0
        label_ex = ""
        nav_box = state["person_lock"] if state["person_lock"] is not None else state["face_lock"]
        # Xác định action bằng hàm riêng
        if nav_box is not None:
            found = True
            x1,y1,x2,y2 = nav_box
            cx = (x1+x2)*0.5; cy=(y1+y2)*0.5
            x_norm_raw = (cx / W)*2.0 - 1.0
            state["x_smooth"] = (1-self.SMOOTH)*state["x_smooth"] + self.SMOOTH*x_norm_raw
            x_norm = state["x_smooth"]
            dist_est = None
            if state["face_lock"] is not None:
                fx1,fy1,fx2,fy2 = state["face_lock"]
                dist_est = self.estimate_dist_from_face(fy2-fy1, H, k_ref=0.18)
                label_ex = "FACE-LOCK"
            else:
                dist_est = self.estimate_dist_from_person(y2-y1, H, k_ref=1.9)
                label_ex = "PERSON-LOCK"
            ratio = (y2-y1)/H
            speed_pct = self.speed_bucket_from_ratio(ratio)
            action = self.get_action_from_state(x_norm, speed_pct, nav_box)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame, f"{label_ex} x={x_norm:.2f} sp={speed_pct}", (x1,max(0,y1-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            cv2.circle(frame,(int(cx),int(cy)),6,(0,0,255),2)
        else:
            action = self.get_action_from_state(0.0, 0, None)
            cv2.putText(frame, "LOST: searching...", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        print(f"[TRACKING ACTION] {action}")
        state["action"] = action
        if state["face_lock"] is not None and nav_box is not state["face_lock"]:
            x1,y1,x2,y2 = state["face_lock"]
            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,200,0),2)
            cv2.putText(frame, "FACE", (x1,max(0,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,200,0), 2)
        if state["person_lock"] is not None and nav_box is not state["person_lock"]:
            x1,y1,x2,y2 = state["person_lock"]
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,200,255),2)
            cv2.putText(frame, "PERSON", (x1,max(0,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,255), 2)
        return frame

    def run(self):
        cap = cv2.VideoCapture(self.CAM_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = self.process_frame(frame)
            cv2.imshow("Face->Person ID Lock & Handoff", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = HumanTracking()
    tracker.run()
