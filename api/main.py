from fastapi import FastAPI, File ,UploadFile
#from fastapi.middleware.cors import CORSMiddlewares
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests
app=FastAPI()


MODEL= tf.keras.models.load_model("model/1")
CLASS_NAMES=['Early Blind','Late Blind', 'Healthy']


@app.get("/ping")
async def ping():
    return "hello, there"

def read_file_as_image(data)-> np.ndarray:
    Image.open(BytesIO(data))


@app.post("/predict")
async def predict(
    file: UploadFile= File(...)

):
    image= read_file_as_image(await file.read())
    img_batch=np.expand_dims(image,0)
    prediction=MODEL.predict(img_batch)

    predicted_class=np.argmax(prediction[0])

    confidence= np.max(prediction[0])

if __name__=="__main__":
    uvicorn.run(app,host='localhost',port=3000)