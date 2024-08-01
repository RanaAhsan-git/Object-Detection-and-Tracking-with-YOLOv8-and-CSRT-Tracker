# Object-Detection-and-Tracking-with-YOLOv8-and-CSRT-Tracker

This repository contains code for detecting and tracking objects using a custom-trained YOLOv8 model and the CSRT tracker. The YOLOv8 model is used to detect objects in the initial frame, and the CSRT tracker is used to track the detected object across subsequent frames. This setup is particularly useful for real-time object tracking using a webcam.

**Features**

Custom-trained YOLOv8 model for object detection.
CSRT tracker for robust object tracking.
Real-time processing using a webcam.
Displays bounding boxes, class names, and confidence scores on the tracked object.

**Requirements**

**To run this code, you need the following libraries:**
OpenCV
Ultralytics YOLO

**You can install these libraries using the following commands:**
sh
Copy code
pip install opencv-python-headless
pip install ultralytics

**How It Works**
Model Training: A custom YOLOv8 model is trained on a dataset. The resulting model weights (.pt file) are used for object detection.
Webcam Initialization: The code captures video from the webcam.
Initial Object Detection: YOLOv8 detects objects in the first frame captured from the webcam.
Tracker Initialization: The CSRT tracker is initialized with the bounding box of the detected object.
Real-Time Tracking: The CSRT tracker updates the object's position in subsequent frames, drawing bounding boxes and displaying class names and confidence scores.
Usage
**Clone this repository**:
sh
Copy code
git clone https://github.com/yourusername/your-repo.git
cd your-repo

**Make sure you have installed the required libraries**

Copy code
pip install opencv-python-headless
pip install ultralytics

**Key Points**
The script initializes the webcam and reads frames in a loop.
It performs object detection using YOLOv8 on the first frame.
If an object is detected, it initializes the CSRT tracker with the detected object's bounding box.
In subsequent frames, it updates the tracker and draws the bounding box, class name, and confidence score.
The script displays the tracking result in a window and exits if the 'q' key is pressed.
Notes
Ensure you have the correct path to the trained YOLOv8 model in the model initialization.
This code assumes the model's class names are accessible through model.names.
Feel free to fork this repository, contribute, and improve the code. Happy coding!
