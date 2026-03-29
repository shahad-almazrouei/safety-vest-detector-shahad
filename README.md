# Safety Vest Detection App

A Dockerized Flask web application for object detection using a fine-tuned YOLOv8 model.

## Overview

This project deploys a previously trained **YOLOv8n** object detection model as a simple web application. The model analyzes an uploaded image and detects two classes:

- **Safety Vest**
- **No Safety Vest**

The app was built as part of a model deployment challenge requiring:
- selection of a previously developed model,
- a minimal Flask interface,
- Docker-based containerization,
- and deployment readiness. 

Based on the training notebook, the model was trained on a custom safety vest dataset and evaluated on a validation split, achieving approximately:
- **Precision:** 0.784
- **Recall:** 0.693
- **mAP@0.5:** 0.733
- **mAP@0.5:0.95:** 0.309

---

## What the model does

The model takes an input image and identifies people wearing:
- a **Safety Vest**
- or **No Safety Vest**

After inference, the application:
1. saves the uploaded image,
2. runs object detection using the trained `best.pt` model,
3. draws bounding boxes on detected objects,
4. displays the annotated image,
5. shows a short prediction summary with detected class counts.

This allows the model to be used through a browser instead of only from a notebook.

---

## Project structure

```text
safety-vest-detector/
├── app.py
├── best.pt
├── requirements.txt
├── Dockerfile
├── README.md
├── deployment_link.txt
├── static/
│   ├── uploads/
│   └── results/
└── templates/
    └── index.html
```

## How to run locally using Docker
1. Open a terminal in the project folder
2. Build the Docker image: docker build -t safety-vest-app .
3. Run the Docker container: docker run -p 5000:5000 safety-vest-app
4. Open the app in your browser: Go to: http://127.0.0.1:5000
   
If the container starts successfully, the app should load in your browser and be ready for testing.

## How to use the interface
1. Open the application in your browser
2. Click Choose File
3. Select an image from your device
4. Click Run Detection
5. View:
    the original uploaded image,
    the detection result image with bounding boxes,
    the prediction summary

The app stores:
the original image in static/uploads
the annotated result image in static/results

## Known issues and limitations
- The model performs better when subjects are clearly visible and large enough in the image.
- Detection performance may decrease for: small or distant people, partially occluded subjects, poor lighting conditions, backgrounds with colors similar to safety   vests.
- The lower mAP@0.5:0.95 compared to mAP@0.5 suggests that bounding box localization is less accurate under stricter evaluation thresholds.
- Because the training dataset was relatively small before augmentation, the model may not generalize perfectly to all real-world scenes.
- The current interface supports image upload only. It does not support video or live webcam detection.

## App Preview
<img width="1297" height="860" alt="image" src="https://github.com/user-attachments/assets/344feebb-eae8-4f26-a681-8caed958223e" />
