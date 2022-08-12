
from fastapi import FastAPI, File, UploadFile
from starlette.responses import RedirectResponse
from io import BytesIO
import io
import string
import time
import os
import numpy as np
import tensorflow as tf
import uvicorn
import cv2   
from PIL import Image
from tensorflow.keras.applications.imagenet_utils import decode_predictions

model = tf.keras.models.load_model(r'C:\Users\shoti\Desktop\Image Processing Practicals\Assignment\Flower Recognition Bot\vgg19')
app = FastAPI(title='Tensorflow FastAPI Starter Pack')

def load_model():
    model = tf.keras.models.load_model(r'C:\Users\shoti\Desktop\Image Processing Practicals\Assignment\Flower Recognition Bot\vgg19')
    print("Model loaded")
    return model


def predict(image: Image.Image):
    global model
    if model is None:
        model = load_model()
    size = (224,224)
    X = []
    image = np.asarray(image.resize(size))
    X.append(image)
    X = np.array(X)
    X = X/255

    return model.predict(X)


def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image

   
@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs")

@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    pred= predict(image)
    pred_digit=np.argmax(pred,axis=1)[0]
    if pred_digit == 0:
        return "Daisy"
    elif pred_digit == 1:
        return "Dandelion"
    elif pred_digit == 2:
        return "Rose"
    elif pred_digit == 3:
        return "Sunflower"
    else:
        return "Tulip"
    



if __name__ == "__main__":
    uvicorn.run(app, debug=True)