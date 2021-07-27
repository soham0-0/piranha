from fastapi import FastAPI
import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from fastapi.middleware.cors import CORSMiddleware
from train import clean_data

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

loaded_model = joblib.load("model/spam_classifier")
loaded_cvec = joblib.load("model/count_vectorizer")
loaded_tfidf_transformer = joblib.load("model/tfidf_transformer")

@app.get("/{content}")
def get_prediction(content: str):
    [content] = clean_data({"message": content})
    return loaded_model.predict(
        loaded_tfidf_transformer.transform(
            loaded_cvec.transform(
                [content]
            )
        )
    )[0]
    