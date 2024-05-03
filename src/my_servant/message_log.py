from enum import Enum


class Role(Enum):
    """Role of the message sender."""

    USER = "user"
    ASSISTANT = "assistant"


class MessageLog:
    """Log of messages exchanged between the user and the assistant."""

    messages: list[dict[str, str]] = []

    def push_new_message(self, role: Role, text: str) -> None:
        """Push a new message to the log.

        :param role: Role of the message sender.
        :param text: Content of the message.
        """
        self.messages.append({"role": role.value, "content": text})

    def add_to_last_message(self, text: str) -> None:
        """Add text to the last message in the log.

        :param text: Text to be added to the last message.
        """
        self.messages[-1]["content"] += text

    def get_messages(self):
        """Get the messages in the log."""
        return self.messages
