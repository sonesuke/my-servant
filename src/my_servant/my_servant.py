from chat_agent import ChatAgent
from voice_agent import VoiceAgent


if __name__ == "__main__":
    chat_agent = ChatAgent()
    voice_agent = VoiceAgent()
    while True:
        text = input("Enter text: ")
        if text == "q":
            break
        responses = chat_agent.chat(text)
        for chunk in responses:
            print(chunk)
            try:
                voice_agent.talk(chunk)
            except Exception as e:
                print(e)
