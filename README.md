# Object-Detection-and-Tracking-with-YOLOv8-and-CSRT-Tracker

# Introduction

This repository contains code for detecting and tracking objects using a custom-trained YOLOv8 model and the CSRT tracker. The YOLOv8 model is used to detect objects in the initial frame, and the CSRT tracker is used to track the detected object across subsequent frames. This setup is particularly useful for real-time object tracking using a webcam.

# **Features**

Custom-trained YOLOv8 model for object detection.

CSRT tracker for robust object tracking.

Real-time processing using a webcam.

Displays bounding boxes, class names, and confidence scores on the tracked object.

# Labeling Dataset with LabelImg

LabelImg is an open-source graphical image annotation tool that is used to label objects in images, typically for object detection tasks. It supports various annotation formats such as YOLO and Pascal VOC.

**Installing LabelImg**

LabelImg can be installed using Python and is compatible with Windows, macOS, and Linux. 

**Install LabelImg:**

You can install LabelImg via pip. Open a terminal or command prompt and run:
```
pip install labelimg
```

Alternatively, you can install LabelImg directly from the source if you need the latest version or wish to contribute to development:
```
git clone https://github.com/tzutalin/labelImg.git
cd labelImg
pip install -r requirements/requirements.txt
python labelImg.py
```
# Using LabelImg to Label Your Dataset

**Open LabelImg:**

After installation, you can start LabelImg by running:
```
labelimg
````
If you installed from the source, run:
```
python labelImg.py
```
**Set Up Directories:**

**Open Directory:** Click on "Open Dir" to select the directory where your images are stored.

**Save Directory:** Click on "Change Save Dir" to select the directory where you want to save your annotations.

# **Label Images:**

**Create Annotation:** Click on the “Create RectBox” button (or press W) to start annotating. Draw a rectangle around the object you want to label.

**Enter Label:** After drawing the rectangle, a dialog will appear asking you to enter a label. Type the name of the object (e.g., "cat", "dog", "car").

**Save Annotations:** The annotations are automatically saved in the format you selected (YOLO or Pascal VOC). You can change the format by going to “View” > “Change Output Format” and selecting your preferred format.

**Navigate Through Images:** Use the arrow keys or the navigation buttons to move to the next or previous image in the folder.

**Finish Labeling:**
Continue labeling each image in your dataset. Once done, all annotations will be saved in the specified directory.

# Annotation Formats
**YOLO Format:**

Annotations are saved in .txt files, where each line represents an object in the format: <class_id> <x_center> <y_center> <width> <height>.

Coordinates are normalized to [0, 1].

# Launch LabelImg.
Open the directory containing your images.

Choose a save directory for annotations.

Start annotating each image by drawing bounding boxes around the objects and assigning labels.

Save annotations in your desired format.

# Additional Tips
**Class Names:** Maintain a consistent list of class names for labeling.

**Consistency:** Ensure annotations are accurate and consistent across your dataset for better model performance.

By following these steps, you can efficiently label your dataset for training object detection models using tools like YOLO.

# Training a Dataset with YOLOv8

# 1. Prepare Your Dataset**

Before training, you need to prepare your dataset. YOLOv8 requires a specific format for annotations, which can be generated using tools like LabelImg.

**Dataset Structure**

**YOLOv8 typically requires the following directory structure:**
```
/dataset
    /images
        /train
            image1.jpg
            image2.jpg
            ...
        /val
            image1.jpg
            image2.jpg
            ...
    /labels
        /train
            image1.txt
            image2.txt
            ...
        /val
            image1.txt
            image2.txt
            ...
```

**Images:** JPEG or PNG files for training and validation.

**Labels:** YOLO format text files (.txt) for annotations where each line contains:

<class_id> <x_center> <y_center> <width> <height>

Coordinates are normalized to [0, 1].
Example of YOLO Format (.txt)
```
0 0.5 0.5 0.2 0.3
1 0.7 0.8 0.1 0.2
```
Here, 0 and 1 are class IDs, and the following values are normalized coordinates.

# 2. Install YOLOv8

YOLOv8 can be installed via pip. Make sure you have Python installed, then use:
```
pip install ultralytics
```
# 3. Prepare Configuration File

You need to create a configuration file to specify your dataset paths and hyperparameters. Create a YAML file (e.g., data.yaml) with the following content:
```
path: /path/to/your/dataset  # Path to your dataset
train: images/train
val: images/val

nc: 2  # Number of classes
names: ['class1', 'class2']  # List of class names
```
Replace /path/to/your/dataset with the actual path to your dataset directory, adjust nc (number of classes), and provide the names of your classes.

# 4. Train the Model

With YOLOv8 installed, you can train your model using the following command:
```
yolo train model=yolov8n.pt data=data.yaml epochs=50 imgsz=640
```
**Parameters Explained:**

**model=yolov8n.pt:** The pre-trained YOLOv8 model to start with (YOLOv8 has different versions like yolov8n for nano, yolov8s for small, etc.).

**data=data.yaml:** Path to your YAML configuration file.

**epochs=50:** Number of training epochs.

**imgsz=640:** Image size (resolution) for training.

You can adjust these parameters based on your needs and available computational resources.

# 5. Monitor Training

The training process will generate logs and save checkpoints. Monitor the output for metrics like loss, precision, recall, and mAP (mean Average Precision). You can visualize the training progress using tools like TensorBoard or directly from the log files.

# 6. Evaluate and Test

After training, evaluate the model on your validation set to check its performance. YOLOv8 will save the best model weights based on the validation metrics.

# 7. Inference

To run inference on new images or videos, use the trained model with the following command:
```
yolo predict model=path/to/best_model.pt source=path/to/image_or_video
```

Replace path/to/best_model.pt with the path to your trained model and path/to/image_or_video with the path to the image or video file you want to test.

**Example Command for Training:**
```
yolo train model=yolov8s.pt data=data.yaml epochs=100 imgsz=640
```

This command trains the yolov8s model for 100 epochs with images resized to 640x640 pixels.

# Additional Tips for Yolo Training
**Data Augmentation:** Use data augmentation techniques to improve model robustness and generalization.

**Hyperparameter Tuning:** Adjust learning rates, batch sizes, and other hyperparameters based on your dataset and hardware.

**Pre-trained Models:** Using pre-trained models as a starting point can significantly reduce training time and improve performance.

By following these steps, you should be able to train a YOLOv8 model on your custom dataset and deploy it for object detection tasks.

# **Requirements for the project**

**To run this code, you need the following libraries:**

**OpenCV
Ultralytics YOLO**

**You can install these libraries using the following commands:**
```
$ pip install opencv-python-headless

$ pip install ultralytics
```

# **How It Works**

**Model Training:** A custom YOLOv8 model is trained on a dataset. The resulting model weights (.pt file) are used for object detection.

**Webcam Initialization:** The code captures video from the webcam.

**Initial Object Detection:** YOLOv8 detects objects in the first frame captured from the webcam.

**Tracker Initialization:** The CSRT tracker is initialized with the bounding box of the detected object.

**Real-Time Tracking:** The CSRT tracker updates the object's position in subsequent frames, drawing bounding boxes and displaying class names and confidence scores.

# Flowchart of process

![flowchart](https://github.com/user-attachments/assets/d8e84a91-f61d-4b56-9eb5-2c1745cb3ea8)


# **Make sure you have installed the required libraries**

```
$ pip install opencv-python-headless

$ pip install ultralytics
```

# **Key Points**

The script initializes the webcam and reads frames in a loop.

It performs object detection using YOLOv8 on the first frame.

If an object is detected, it initializes the CSRT tracker with the detected object's bounding box.

In subsequent frames, it updates the tracker and draws the bounding box, class name, and confidence score.

The script displays the tracking result in a window and exits if the 'q' key is pressed.

# **Notes**

Ensure you have the correct path to the trained YOLOv8 model in the model initialization.

This code assumes the model's class names are accessible through model.names.

For queries mail at **ahsanaslam9990@gmail.com**

# Feel free to fork this repository, contribute, and improve the code. Happy coding!
