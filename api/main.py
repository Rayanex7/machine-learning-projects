from fastapi import FastAPI
from pydantic import BaseModel
import joblib

model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = FastAPI(title="Spam Filter API")

class EmailRequest(BaseModel):
    text: str

@app.post("/predict")
def predict_spam(request: EmailRequest):
    features = vectorizer.transform([request.text])

    prediction = model.predict(features)[0]

    return {
            "email": request.text,
            "prediction": str(prediction)
            }
