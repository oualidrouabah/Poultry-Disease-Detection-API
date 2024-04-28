import numpy as np
from fastapi import FastAPI, UploadFile, File
from io import BytesIO
from keras.preprocessing import image
from keras.models import load_model

# Load the saved Keras model
model = load_model('PoultryModel.h5')

# Define a function to preprocess the input image
def preprocess_image(img):
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Define a function to make predictions
def predict_image(img, model):
    # Preprocess the image
    img = preprocess_image(img)
    # Make prediction
    prediction = model.predict(img)
    return prediction

def interpret_prediction(prediction):
    classes = ['cocci', 'healthy', 'NCD', 'salmo']
    threshold = 0.5
    best_score_index = 0
    best_score = prediction[0][0]  
    for i, score in enumerate(prediction[0]):
        if score > best_score:
            best_score = score
            best_score_index = i
    if best_score >= threshold:
        return classes[best_score_index]
    else:
        return "Unknown"


app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = image.load_img(BytesIO(contents), target_size=(256, 256))
    prediction = predict_image(img, model)
    return {"prediction": interpret_prediction(prediction)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
