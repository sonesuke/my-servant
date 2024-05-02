from voicevox_core import VoicevoxCore
import sounddevice as sd
import numpy as np


class VoiceAgent:
    """VoiceAgent class to convert text to speech."""

    speaker_id: int = 2

    def __init__(self) -> None:
        """Initialize VoiceAgent."""
        self.core = VoicevoxCore(
            open_jtalk_dict_dir="voicevox_core/open_jtalk_dic_utf_8-1.11"
        )
        self.core.load_model(self.speaker_id)
        self.stream = sd.OutputStream(
            samplerate=24000,
            channels=1,
            dtype=np.int16,
        )
        self.stream.start()

    def talk(self, text: str) -> None:
        """Convert text to speech and play it.

        :param text: Text to be converted to speech.
        """
        wav = self.core.tts(text, speaker_id=self.speaker_id)
        wav_array = np.frombuffer(wav, dtype=np.int16)
        self.stream.write(wav_array)


if __name__ == "__main__":
    voice_agent = VoiceAgent()
    voice_agent.talk("こんにちは、私はボイスエージェントです。")
