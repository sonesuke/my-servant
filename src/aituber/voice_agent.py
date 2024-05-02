from voicevox_core import VoicevoxCore
import sounddevice as sd
import numpy as np

speaker_id = 2


class VoiceAgenet:
    def __init__(self) -> None:
        self.core = VoicevoxCore(
            open_jtalk_dict_dir="voicevox_core/open_jtalk_dic_utf_8-1.11"
        )
        self.core.load_model(speaker_id)

    def talk(self, text: str) -> None:
        wav = self.core.tts(text, speaker_id=speaker_id)
        wav_array = np.frombuffer(wav, dtype=np.int16)
        sd.play(wav_array, 24000, blocking=True)


def text_to_voice():
    agent = VoiceAgenet()
    while True:
        text = input("Enter text: ")
        if text == "q":
            return
        agent.talk(text)


if __name__ == "__main__":
    text_to_voice()
