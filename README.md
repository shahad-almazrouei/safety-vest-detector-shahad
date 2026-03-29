# Safety Vest Detection App

This project deploys a YOLOv8 object detection model using Flask and Docker. The model detects two classes:
- Safety Vest
- No Safety Vest

## Model Performance
Based on validation results from training:
- mAP@0.5 = 0.733
- mAP@0.5:0.95 = 0.309

## Features
- Upload an image through a web interface
- Detect safety vest and no safety vest objects
- Display annotated detection results
- Show prediction summary

## Files
- `app.py` - main Flask application
- `templates/index.html` - frontend page
- `best.pt` - trained YOLO model
- `requirements.txt` - required Python packages
- `Dockerfile` - Docker configuration

## Run Locally with Docker
```bash
docker build -t safety-vest-app .
docker run -p 5000:5000 safety-vest-app