import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HandMarker:
    def __init__(self):
        model_path = "./models/hand_landmarker.task"
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            num_hands = 2
        )
        
        self.landmarker = HandLandmarker.create_from_options(options)
        
    def detect_hand_in_frame(self, frame):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        return self.landmarker.detect(mp_image)
    
