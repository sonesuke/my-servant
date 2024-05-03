from chat_agent import ChatAgent
from summarize_agent import SummarizeAgent
from voice_agent import VoiceAgent


if __name__ == "__main__":
    chat_agent = ChatAgent()
    voice_agent = VoiceAgent()
    summarize_agent = SummarizeAgent()
    while True:
        text = input("> ")
        if text == "q":
            break
        if text == "s":
            response = summarize_agent.summarize(chat_agent.message_log)
            print(response)
        else:
            responses = chat_agent.chat(text)
            for chunk in responses:
                print(chunk)
                try:
                    voice_agent.talk(chunk)
                except Exception as e:
                    print(e)
