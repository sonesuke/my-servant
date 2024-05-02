from typing import Generator
import ollama

STOP_CHARACTERS = ["。", "\n"]

class ChatAgent:
    """Chat agent that uses Ollama API to chat with the user."""

    def find_stop_character(self, text: str) -> int:
        """Find the first stop character in the text.

        :param text: Text to be searched.
        :return: Index of the stop character in text.
        """
        for char in STOP_CHARACTERS:
            index = text.find(char)
            if index != -1:
                return index
        return -1

    def chat(self, text: str) -> Generator[str, None, None]:
        """Chat with the user using Ollama API.

        :param text: Text to chat with the user.
        """
        stream = ollama.chat(
            model="suzume-mul",
            messages=[
                {"role": "user", "content": text},
            ],
            stream=True,
        )
        buffer: str = ""
        for chunk in stream:
            buffer += chunk["message"]["content"]
            found = self.find_stop_character(buffer)
            if found != -1:
                sentence = buffer[: found + 1]
                if len(sentence.strip()) > 0:
                    yield sentence
                buffer = buffer[found + 1 :]
        yield buffer
