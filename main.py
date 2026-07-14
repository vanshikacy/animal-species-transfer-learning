from fastapi import FastAPI, UploadFile, File
from PIL import Image
from src.predict import predict_image

app=FastAPI()

@app.get("/")
def home():
    return {"message": "Animal Classification (Transfer Learning Project)"}

@app.post("/predict")
async def predict(file: UploadFile=File(...)):

    image=Image.open(file.file).convert("RGB")
    prediction=predict_image(image)

    return {
        "prediction": prediction
    }