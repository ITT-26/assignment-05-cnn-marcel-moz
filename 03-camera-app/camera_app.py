import cv2, sys, time, keras
import screeninfo
import numpy as np
from hand_marker import HandMarker


class CameraApp:
    def __init__(self, video_id):
        self.cap = cv2.VideoCapture(video_id)
        self.COLOR_CHANNELS = 3
        self.IMG_SIZE = 64
        self.model = None
        self.labels = ["like", "stop", "dislike", "peace"]
        self.countdown_time = 3
        self.countdown_run_time = 0
        self.counting_down = False
        self.is_running = False
        monitors = screeninfo.get_monitors()
        self.display_width = monitors[0].width
        self.display_height = monitors[0].height
        self.background = None
        self.label_names = ["like", "stop", "dislike", "peace"]
        self.hand_marker = HandMarker()
        self.PRED_THRESHHOLD = 0.825
        self.filter_active = False
        self.gesture_interval = 2
        self.zoom_interval = 0.5

    def setup_cap(self):
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, sys.maxsize)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, sys.maxsize)

    def cap_background(self):
        self.background = cv2.cvtColor(self.get_frame(), cv2.COLOR_BGR2GRAY)

    def get_frame(self):
        ret, frame = self.cap.read()
        return frame

    def get_dimensions_from_hand_landmarks(self, landmarks, frame):
        x_values = [lm.x for lm in landmarks]
        y_values = [lm.y for lm in landmarks]

        frame_height, frame_width = frame.shape[:2]

        x_min = int(min(x_values) * frame_width)
        y_min = int(min(y_values) * frame_height)
        x_max = int(max(x_values) * frame_width)
        y_max = int(max(y_values) * frame_height)

        padding = int(frame_height * 0.05)

        return (
            max(0, x_min - padding),
            max(0, y_min - padding),
            min(frame_width, x_max + padding),
            min(frame_height, y_max + padding),
        )

    def get_hand_boxes(self, frame):

        detection = self.hand_marker.detect_hand_in_frame(frame)
        hand_boxes = []
        for landmarks in detection.hand_landmarks:
            x_min, y_min, x_max, y_max = self.get_dimensions_from_hand_landmarks(
                landmarks, frame
            )
            f_copy = frame.copy()

            cropped = f_copy[y_min:y_max, x_min:x_max]
            hand_boxes.append(cropped)

        return hand_boxes

    def prepare_hand_image(self, hand_image):
        resized = cv2.resize(hand_image, (self.IMG_SIZE, self.IMG_SIZE))
        resized = np.array(resized).astype("float32")
        resized = resized / 255.0
        reshaped = resized.reshape(
            -1, self.IMG_SIZE, self.IMG_SIZE, self.COLOR_CHANNELS
        )

        return reshaped

    def create_model(self):
        self.model = keras.models.load_model("./models/gesture_recognition.keras")

    def train_model(self):
        pass

    def predict_frame(self, gesture_img):
        if self.model is None or gesture_img is None:
            return
        return self.model.predict(gesture_img, verbose=0)

    def zoom_in(self):
        pass

    def zoom_out(self):
        pass

    def add_filter(self, frame):
        pass

    def setup(self):
        self.setup_cap()
        self.cap_background()
        self.create_model()
        self.train_model()

    def perform_countdown(self):
        pass

    def run(self):
        frame = self.get_frame()
        if frame is None:
            return
        frame = cv2.flip(frame, 1)
        # flip so camere is more inuitive

        hand_imgs = self.get_hand_boxes(frame.copy())

        for hand_img in hand_imgs:
            prepared = self.prepare_hand_image(hand_img)
            prediction = self.predict_frame(prepared)
            max_pred = np.max(prediction)
            if max_pred > self.PRED_THRESHHOLD:
                print(self.label_names[np.argmax(prediction)], max_pred)
                # print(prediction)
                # cv2.imshow("hand", hand_img)
                gesture = self.label_names[np.argmax(prediction)]
                if not self.counting_down and gesture == "stop":
                    self.perform_countdown()

                if gesture == "like":
                    self.zoom_in()

                if gesture == "dislike":
                    self.zoom_out()

                if gesture == "peace":
                    self.filter_active = True

        display_frame = frame.copy()

        height, width = display_frame.shape[:2]

        if height > self.display_height or width > self.display_width:
            display_frame = cv2.resize(
                display_frame, (self.display_width, self.display_height)
            )

        if self.filter_active:
            display_frame = self.add_filter(display_frame)

        cv2.imshow("Camera App", display_frame)
        time.sleep(0.01)


def main():
    video_id = 0

    if len(sys.argv) > 1:
        video_id = int(sys.argv[1])

    app = CameraApp(video_id)
    app.setup()

    start_time = time.time()
    last_background_time = start_time
    BACKGROUND_INTERVAL = 1

    while True:
        run_time = time.time() - start_time
        if time.time() - last_background_time > BACKGROUND_INTERVAL:
            # alle paar sekunden neuer background (für größere bewegungen)
            app.cap_background()
            last_background_time = time.time()

        app.run()
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:  # q or ESC
            break


if __name__ == "__main__":
    main()
