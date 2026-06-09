import cv2
import sys
import time
import keras
import math
import screeninfo
import argparse
import numpy as np
from hand_marker import HandMarker
from collections import deque
from pathlib import Path


class CameraApp:

    def __init__(self, video_id, timer, path, filename):
        self.cap = cv2.VideoCapture(video_id)
        self.path = path
        self.filename = Path(filename)
        self.version_counter = 1
        self.stem = self.filename.stem
        self.COLOR_CHANNELS = 3
        self.IMG_SIZE = 64
        self.model = None
        self.labels = ["like", "stop", "dislike", "peace"]
        self.countdown_time = timer
        self.countdown_end_time = 0
        self.is_counting_down = False
        self.is_running = False
        monitors = screeninfo.get_monitors()
        self.display_width = monitors[0].width
        self.display_height = monitors[0].height
        self.label_names = ["like", "stop", "dislike", "peace"]
        self.hand_marker = HandMarker()
        self.PRED_THRESHHOLD = 0.9
        self.filter_active = False
        self.peace_interval = 1
        self.last_peace_time = 0
        self.zoom_factor = 1
        self.zoom_step = 0.01
        self.gesture_deque = deque(maxlen=3)
        self.gesture_result = None
        self.FILTER_AMPLITUDE = 30
        self.FILTER_SMOOTHING = 50

    def setup_cap(self):
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, sys.maxsize)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, sys.maxsize)

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

        padding = int(frame_width * 0.025)

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
            # cv2.imshow("test", cropped)

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
        self.model = keras.models.load_model(
            "./models/gesture_recognition.keras")

    def predict_frame(self, gesture_img):
        if self.model is None or gesture_img is None:
            return
        return self.model.predict(gesture_img, verbose=0)

    def zoom_in(self):
        if self.zoom_factor < 4:
            self.zoom_factor += self.zoom_step

    def zoom_out(self):
        if self.zoom_factor > 1:
            self.zoom_factor -= self.zoom_step

    def apply_zoom(self, frame):
        if self.zoom_factor == 1:
            return frame  # shortcut when not zoomed

        height, width = frame.shape[:2]

        center_y = height // 2
        center_x = width // 2

        new_width = int(width / self.zoom_factor)
        new_height = int(height / self.zoom_factor)

        new_x = center_x - new_width // 2
        new_y = center_y - new_height // 2

        cropped = frame[new_y: new_y + new_height, new_x: new_x + new_width]

        return cv2.resize(cropped, (width, height))

    def apply_filter(self, frame, runtime=0):

        height, width = frame.shape[:2]

        map_x = np.zeros((height, width), np.float32)
        map_y = np.zeros((height, width), np.float32)
        # create distortion maps
        # from https://stackoverflow.com/questions/59776772/python-opencv-how-to-apply-radial-barrel-distortion

        normal_y = np.arange(height)
        # use arange for better performence than with nested loop

        for x in range(width):

            map_x[:, x] = x

            wave = self.FILTER_AMPLITUDE * \
                math.cos(x / self.FILTER_SMOOTHING + runtime)
            # SMOOTHING constant was suggest but ChatGPT for fixing the filter that produced only random pattern

            map_y[:, x] = normal_y + wave

        return cv2.remap(frame, map_x, map_y,
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    def setup(self):
        self.setup_cap()
        self.create_model()

    def start_countdown(self, starttime):
        self.is_counting_down = True
        self.countdown_end_time = starttime + self.countdown_time

    def stop_countdown(self):
        self.is_counting_down = False

    def set_gesture_result(self):
        if len(self.gesture_deque) == 3:
            if self.gesture_deque[0] == self.gesture_deque[1] == self.gesture_deque[2]:
                if self.gesture_deque[0] != "":
                    self.gesture_result = self.gesture_deque[0]
                    return
        self.gesture_result = None

    def get_result_file_with_version(self):

        while (self.path / self.filename).exists():
            self.filename = Path(
                self.stem + str(self.version_counter) + self.filename.suffix
            )
            self.version_counter += 1
        return self.path / self.filename

    def save_image(self, frame):
        self.path.mkdir(parents=True, exist_ok=True)
        final_file_name = self.get_result_file_with_version()
        cv2.imwrite(str(final_file_name), frame)

    def run(self, runtime):
        frame = self.get_frame()
        if frame is None:
            return
        frame = cv2.flip(frame, 1)
        # flip so camere is more inuitive

        display_frame = frame.copy()

        og_height, og_width = display_frame.shape[:2]

        if og_height > self.display_height or og_width > self.display_width:
            display_frame = cv2.resize(
                display_frame, (self.display_width, self.display_height)
            )

        hand_imgs = self.get_hand_boxes(frame.copy())

        if len(hand_imgs) == 0:
            self.gesture_deque.append("")

        if not self.is_counting_down:
            for hand_img in hand_imgs:
                prepared = self.prepare_hand_image(hand_img)
                prediction = self.predict_frame(prepared)
                max_pred = np.max(prediction)

                if max_pred > self.PRED_THRESHHOLD:
                    gesture = self.label_names[np.argmax(prediction)]

                    self.gesture_deque.append(gesture)
                else:
                    self.gesture_deque.append("")

        self.set_gesture_result()

        if not self.is_counting_down and self.gesture_result == "stop":
            self.start_countdown(runtime)
        if self.gesture_result == "like":
            self.zoom_in()

        if self.gesture_result == "dislike":
            self.zoom_out()

        if self.gesture_result == "peace" and runtime - self.last_peace_time > self.peace_interval:
            if self.filter_active:
                self.filter_active = False
            else:
                self.filter_active = True

            self.last_peace_time = runtime

        display_frame = self.apply_zoom(display_frame)

        if self.filter_active:
            display_frame = self.apply_filter(display_frame, runtime)

        if self.is_counting_down:
            countdown = math.ceil(self.countdown_end_time - runtime)

            height, width = display_frame.shape[:2]

            if countdown <= 0:
                self.stop_countdown()

                output_frame = display_frame.copy()
                self.save_image(output_frame)

                cv2.rectangle(display_frame, (0, 0),
                              (width, height),
                              (0, 0, 255),
                              10
                              )

            else:
                x_text, y_text,  =  50, int(height * 0.25 * self.zoom_factor)

                cv2.putText(display_frame, str(countdown), (x_text, y_text),
                            cv2.FONT_HERSHEY_DUPLEX, 16, (0, 0, 0), 10)

        cv2.imshow("Camera App", display_frame)
        time.sleep(0.015)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--timer", default=3, type=int,
                        help="countdown after triggering the camera app")
    parser.add_argument("--path", default="./photos/",
                        help="output path for taken images")
    parser.add_argument("--fname", default="image",
                        help="output file name without suffix (optional)")
    parser.add_argument("--video_id", default=0, type=int,
                        help="video_id of your camera in case you need to change it")

    args = parser.parse_args()

    video_id = args.video_id
    timer = args.timer
    path = Path(args.path)
    filename = args.fname + ".png"

    app = CameraApp(video_id, timer, path, filename)
    print("Starting...")
    app.setup()

    start_time = time.time()

    while True:
        run_time = time.time() - start_time
        app.run(run_time)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:  # q or ESC
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    main()
