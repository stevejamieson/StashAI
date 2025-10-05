import json
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

class ChatModel:
    def __init__(self, intents_file="intents.json"):
        with open(intents_file) as f:
            data = json.load(f)

        self.intents = data["intents"]
        self.vectorizer = CountVectorizer()
        self.model = LogisticRegression()

        X, y = [], []
        for intent in self.intents:
            for pattern in intent["patterns"]:
                X.append(pattern.lower())
                y.append(intent["tag"])

        X_vec = self.vectorizer.fit_transform(X)
        self.model.fit(X_vec, y)

    def get_response(self, user_input):
        input_vec = self.vectorizer.transform([user_input.lower()])
        tag = self.model.predict(input_vec)[0]
        for intent in self.intents:
            if intent["tag"] == tag:
                return random.choice(intent["responses"])
