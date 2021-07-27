import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import svm 
from sklearn.metrics import classification_report,confusion_matrix
import joblib

def clean_data(dataframe):
    ps = PorterStemmer()
    corpus = []
    for i in range(0, len(dataframe)):
        review = re.sub('[^a-zA-Z]', ' ', dataframe['message'][i])
        review = review.lower()
        review = review.split()
        
        review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
        review = ' '.join(review)
        corpus.append(review)
    return corpus


if __name__ == "__main__":
    data = pd.read_csv('dataset/spam', sep="\t", names=["label", "message"])
    print(data.describe())

    X = clean_data(data)
    Y = data["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state = 50)
    
    count_vectorizer = CountVectorizer().fit(X_train)
    X_transform = count_vectorizer.transform(X_train)

    tfidf_transformer = TfidfTransformer().fit(X_transform)
    X_tfidf = tfidf_transformer.transform(X_transform)
    
    tuned_parameters = {'kernel': ['rbf','linear'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}
    model = GridSearchCV(svm.SVC(), tuned_parameters).fit(X_tfidf, y_train)

    x_test_transformed = tfidf_transformer.transform(count_vectorizer.transform(X_test))
    predictions = model.predict(x_test_transformed)
    
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test,predictions))

    joblib.dump(model, 'model/spam_classifier')
    joblib.dump(count_vectorizer, 'model/count_vectorizer')
    joblib.dump(tfidf_transformer, 'model/tfidf_transformer');