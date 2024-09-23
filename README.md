# YOLOv8 Object Detection API

This repository contains a FastAPI-based API service for object detection using a pre-trained YOLOv8 model. The service can be deployed inside a Docker container.

## Features

1. **Object Detection**: Uses a pre-trained YOLOv8 model to detect objects in images.
2. **API Endpoints**:
   - `/predictClassesJson/`: Returns predictions classes (bounding boxes with categories) for the uploaded image.
   - `/predictClassesOnImage/`: Returns the processed image with bounding boxes for predicted classes drawn.

## Requirements

- Python 3.10+
- FastAPI
- Uvicorn
- Ultralytics
- Pillow
- Docker
- python-multipart
- aiofiles

## Installation and testing manual

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/yolov8-fastapi.git
   cd yolov8-fastapi
2. **Open Docker Deskop**
3. **Build docker image**
   docker build -t yolo-predict-server .
4. **Run server application**
   docker run -p 8000:8080 yolo-predict-server
