import unittest
import os
from model import ChatModel

class TestChatModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Use a test-specific intents file and model path
        cls.chatbot = ChatModel(intents_path="test_intents.json", model_path="test_model.pkl")

    def test_intents_loaded(self):
        self.assertTrue("intents" in self.chatbot.intents)
        self.assertGreater(len(self.chatbot.intents["intents"]), 0)

    def test_model_trains_and_predicts(self):
        response = self.chatbot.get_response("Hello")
        self.assertIsInstance(response, str)
        self.assertIn(response, [r for i in self.chatbot.intents["intents"] if i["tag"] == "greeting" for r in i["responses"]])

    def test_unknown_response(self):
        response = self.chatbot.get_response("asldkfjasldkfj")
        self.assertIn(response, self.chatbot.intents["intents"][-1]["responses"])  # assumes 'unknown' is last

    @classmethod
    def tearDownClass(cls):
        # Clean up test model file
        if os.path.exists("test_model.pkl"):
            os.remove("test_model.pkl")

if __name__ == "__main__":
    unittest.main()