from fastapi import FastAPI
import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

dataframe = pd.read_csv("dataset/spam.csv")

x = dataframe["EmailText"]
y = dataframe["Label"]
x_train,y_train = x[0:4457],y[0:4457]
x_test,y_test = x[4457:],y[4457:]

cv = CountVectorizer()
features = cv.fit_transform(x_train)
model = joblib.load('model/spam_classifier')

app = FastAPI()

@app.get("/{content}")
def get_prediction(content: str):
    return model.predict(cv.transform([content]))[0]
    