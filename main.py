from fastapi import FastAPI, Query
from pydantic import BaseModel
from model import ChatModel

app = FastAPI()
chatbot = ChatModel()

class Message(BaseModel):
    text: str

@app.post("/chat")
def chat(message: Message):
    response = chatbot.get_response(message.text)
    return {"response": response}
