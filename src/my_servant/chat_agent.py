from typing import Generator
import ollama
from message_log import MessageLog, Role
from prompts import SYSTEM_PROMPT


class ChatAgent(object):
    """Chat agent that uses Ollama API to chat with the user."""

    def __init__(self):
        self.message_log = MessageLog()

    def find_stop_character(self, text: str) -> int:
        """Find the first stop character in the text.

        :param text: Text to be searched.
        :return: Index of the stop character in text.
        """
        STOP_CHARACTERS = ["。", "\n"]
        for char in STOP_CHARACTERS:
            index = text.find(char)
            if index != -1:
                return index
        return -1

    def chat(
        self, text: str, prompt: str = SYSTEM_PROMPT
    ) -> Generator[str, None, None]:
        """Chat with the user using Ollama API.

        :param text: Text to chat with the user.
        """
        self.message_log.push_new_message(Role.USER, text)
        stream = ollama.chat(
            model="suzume-mul",
            messages=[{"role": "system", "content": prompt}]
            + self.message_log.get_messages(),
            stream=True,
        )
        buffer: str = ""
        self.message_log.push_new_message(Role.ASSISTANT, "")
        for chunk in stream:
            buffer += chunk["message"]["content"]
            found = self.find_stop_character(buffer)
            if found != -1:
                sentence = buffer[: found + 1]
                if len(sentence.strip()) > 0:
                    self.message_log.add_to_last_message(sentence)
                    yield sentence
                buffer = buffer[found + 1 :]

        if len(buffer.strip()) > 0:
            self.message_log.add_to_last_message(buffer)
            yield buffer
