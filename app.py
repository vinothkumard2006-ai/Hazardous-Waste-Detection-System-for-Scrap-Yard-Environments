from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
from PIL import Image
import io
import os



# LOAD TRAINED MODEL
# -----------------------------
MODEL_PATH = "runs/train_custom-3/weights/best.pt"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = YOLO(MODEL_PATH)




# FASTAPI APP
# -----------------------------
app = FastAPI(
    title="Hazardous Waste Detection API",
    description="Real-time hazardous waste object detection using YOLOv8",
    version="1.0"
)



@app.get("/")
def home():
    return {
        "message": "Hazardous Waste Detection API is running"
    }



@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Read uploaded image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # Run prediction
    results = model.predict(image, conf=0.4)

    detections = []

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])

            detections.append({
                "class": model.names[cls_id],
                "confidence": round(confidence, 4)
            })

    return {
        "filename": file.filename,
        "detections": detections
    }