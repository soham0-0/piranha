from fastapi import FastAPI
import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from fastapi.middleware.cors import CORSMiddleware

dataframe = pd.read_csv("dataset/spam.csv")

cv = CountVectorizer()
features = cv.fit_transform(dataframe["EmailText"][0:4457])
model = joblib.load('model/spam_classifier')

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/{content}")
def get_prediction(content: str):
    return model.predict(cv.transform([content]))[0]
    