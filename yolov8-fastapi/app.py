from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import shutil
import os
import cv2

app = FastAPI()

# Load model
model = YOLO("best.pt")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.get("/")
def home():
    return {"message": "YOLOv8 FastAPI API Running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Save uploaded image
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run prediction
    results = model(file_path)

    detections = []

    for result in results:
        boxes = result.boxes

        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            detections.append({
                "class_id": cls,
                "class_name": model.names[cls],
                "confidence": conf
            })

    return {
        "filename": file.filename,
        "detections": detections
    }