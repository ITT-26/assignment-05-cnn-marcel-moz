[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/cMaQVOgt)

# General
- First, install the requirements from [requirements.txt](requirements.txt).
# Task 1 Exploring Hyperparameters 
- The files for task 1 are in [`./01-hyperparameters/`](./01-hyperparameters/).
- The notebook [`hyperparameters.ipynb`](./01-hyperparameters/hyperparameters.ipynb) contains the approach and  assumptions defined before testing.
- The code for evaluation can also be found in the notebook.
- You might need to change the path to the HaGRID dataset (subset from GRIPS) in the frist cell of the notebook if you want to execute the code.
- The results as well as the discussion, possible explanations and plots are located in [`documentation_results.md`](./01-hyperparameters/documentation_results.md).
- The plot image files can be found in [`./01-hyperparameters/plots/`](./01-hyperparameters/plots/)

# Task 2 Gathering a Dataset 
- The files for task 2 are in [`./02-dataset/`](./02-dataset/).
- The captured images can be found in the subfolders of [`./02-dataset/gesture_dataset/`](./02-dataset/gesture_dataset/). The annotatations file [`annot-marcel.json`](./02-dataset/gesture_dataset/annot-marcel.json) is directly in the folder [`./02-dataset/gesture_dataset/`](./02-dataset/gesture_dataset/).
- Predictions for the captured images were made with the notebook [`gesture_cnn.ipynb`](./02-dataset/gesture_cnn.ipynb). 
- You might need to change the path to the HaGRID dataset (subset from GRIPS) in the frist cell of the notebook if you want to execute the code.
- The resulting confusion matrix is [`conf-matrix.png`](./02-dataset/conf-matrix.png).

# Task 3 Gesture-controlled Camera App

- The files for task 3 are in [`./03-camera-app/`](./03-camera-app/).
- The notebook [`gesture_cnn.ipynb`] was used for training the gesture prediction model [`gesture_recognition.keras`](./03-camera-app/models/gesture_recognition.keras) located in [`./03-camera-app/models/](./03-camera-app/models/).
- The model [`hand_landermarker.task`](./03-camera-app/models/hand_landmarker.task) in the same folder is a model from [`mediapipe`](https://developers.google.com/edge/mediapipe/solutions/guide) for hand detection.
- Run [`camera_app.py](./03-camera-app/camera_app.py) via command line to start the camera app
    - You can use the parameter `--timer` to specify the countdown for taking a photo.
    - You can use the parameter `--path` to specify the output path for taken photos.
    - You can use the parameter `--fname` to specify a file name for your taken images (no filetype).
    - You can use the parameter `--video_id` to specify another id for your webcam in case you need to. 
- For example you can use `python .\camera_app.py --timer 5 --fname mypic --path ./images/ --video_id 0` or simply use `python .\camera_app.py` to use defaults.
- When the camera app is running you can use the gesture `stop` (flat hand, palm facing towards the camera, fingertips pointing upwards) to trigger the countdown and take a photo after the countdown is over.
- After starting the countdown gestures are not detected until the photo is taken, so you can pose however you want.
- You can use `like` (=thumbs up) or `dislike` (=thumps down) to zoom in (=up) or zoom out (=down). Hold the gesture for continuous zooming until the desired zoom is reached.
- You can use `peace` gesture to activate a cool filter. To deactivate the filter simply show the `peace` gesture again.
- You can quit the program by pressing `q` or `ESC`.

