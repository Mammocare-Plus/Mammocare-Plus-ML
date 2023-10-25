from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from constants import CLASS_NAMES, MODEL_PATH
from PIL import Image
import numpy as np
import io
import tensorflow as tf

app = FastAPI()

origins = [
    "http://127.0.0.1:5173",
    "http://localhost:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"Message": "ML Inference microservice"}

@app.post("/predict")
def predict(file: UploadFile = File(...)):
    try:
        print(file.filename)
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        model.compile(tf.keras.optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
        
        file_content = file.file.read()
        img_file = io.BytesIO(file_content)

        img = Image.open(img_file)
        img = img.resize((50,50))

        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        prediction = CLASS_NAMES[np.argmax(predictions.flatten())]

        return JSONResponse(status_code=200, content={"prediction": prediction, "model_name": "CUSTOM CNN"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    
if __name__ == "__main__":
    uvicorn.run("microservice:app", host="127.0.0.1", port=8080, reload=True)