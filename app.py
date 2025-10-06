from flask import Flask, render_template, request
from model import ChatModel

app = Flask(__name__)
chatbot = ChatModel()

@app.route("/", methods=["GET", "POST"])
def index():
    response = ""
    if request.method == "POST":
        user_input = request.form["user_input"]
        response = chatbot.get_response(user_input)
    return render_template("index.html", response=response)

if __name__ == "__main__":
    app.run(debug=True)