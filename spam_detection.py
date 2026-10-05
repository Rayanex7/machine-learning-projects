import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


df = pd.read_csv('emails.csv')

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(df["text"])

X_train, X_test, y_train, y_test = train_test_split(X, df['spam'], test_size=0.2, random_state=42)

model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

email1 = ["CONGRATULATIONS! You have won $1,000,000! Click here now to claim your FREE prize. Limited time offer!"]
email2 = ["Hi John, just wanted to remind you that our meeting is tomorrow at 10 AM. Please bring the project documents."]

test1 = vectorizer.transform(email1)
test2 = vectorizer.transform(email2)

pred1 = model.predict(test1)
pred2 = model.predict(test2)

print("Prediction 1:", pred1)
print("Prediction 2:", pred2)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
