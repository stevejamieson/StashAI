import json
import random
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Download tokenizer
nltk.download('punkt')

# Load intents
with open("intents.json") as f:
    data = json.load(f)

# Prepare training data
X, y = [], []
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        X.append(pattern.lower())
        y.append(intent["tag"])

# Vectorize and train
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)
model = LogisticRegression()
model.fit(X_vec, y)

# Chat loop
def chatbot_response(user_input):
    input_vec = vectorizer.transform([user_input.lower()])
    tag = model.predict(input_vec)[0]
    for intent in data["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

print("🤖 Chatbot is ready! Type 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Bot: Goodbye!")
        break
    print("Bot:", chatbot_response(user_input))
