import json
import random
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

class ChatModel:
    def __init__(self, intents_path="intents.json", model_path="model.pkl"):
        self.intents_path = intents_path
        self.model_path = model_path
        self.intents = self.load_intents()
        self.vectorizer = None
        self.model = None
        self.load_or_train_model()

    def load_intents(self):
        with open(self.intents_path, "r") as f:
            return json.load(f)

    def prepare_training_data(self):
        X, y = [], []
        for intent in self.intents["intents"]:
            for pattern in intent["patterns"]:
                X.append(pattern.lower())
                y.append(intent["tag"])
        return X, y

    def train_model(self):
        X, y = self.prepare_training_data()
        self.vectorizer = CountVectorizer()
        X_vec = self.vectorizer.fit_transform(X)
        self.model = LogisticRegression()
        self.model.fit(X_vec, y)
        with open(self.model_path, "wb") as f:
            pickle.dump((self.vectorizer, self.model), f)

    def load_or_train_model(self):
        try:
            with open(self.model_path, "rb") as f:
                self.vectorizer, self.model = pickle.load(f)
        except FileNotFoundError:
            self.train_model()

    def get_response(self, user_input):
        input_vec = self.vectorizer.transform([user_input.lower()])
        probs = self.model.predict_proba(input_vec)[0]
        max_prob = max(probs)
        tag = self.model.classes_[probs.argmax()]
        # Confidence threshold
        if max_prob < 0.5:  # You can tune this value
            unknown_intent = next((i for i in self.intents["intents"] if i["tag"] == "unknown"), None)
            if unknown_intent:
                return random.choice(unknown_intent["responses"])
            return "I'm not sure how to respond to that."

        for intent in self.intents["intents"]:
            if intent["tag"] == tag:
                return random.choice(intent["responses"])
        # If no matching intent is found (shouldn't happen)
        return "I'm not sure how to respond to that."
