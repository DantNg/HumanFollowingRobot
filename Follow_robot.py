from tracking import HumanTracking
from MotorController import MotorController
import cv2
import time



if __name__ == "__main__":
    tracker = HumanTracking()
    motor = MotorController(port="COM19", baudrate=115200)
    cap = cv2.VideoCapture(tracker.CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            processed = tracker.process_frame(frame)
            action = tracker.state.get("action", "stop")
            # Điều khiển động cơ dựa vào action
            if action == "forward":
                motor.send_if_changed(20, 1, 20, 1)
            elif action == "left":
                motor.send_if_changed(0, 0, 10, 1)
            elif action == "right":
                motor.send_if_changed(10, 1, 0, 0)
            else:
                motor.send_if_changed(0, 1, 0, 1)
            cv2.imshow("Face->Person ID Lock & Handoff", processed)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
