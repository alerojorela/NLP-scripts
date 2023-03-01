from time import time
from pathlib import Path



class Transcriber():
    def transcribe(self):
        pass


class Wave2vecTranscriber(Transcriber):
    def __init__(self, language: str):
        print(
            "Make sure you're providing audio with 16kHz sample rate. The base model is pretrained on 16kHz sampled speech audio.")
        self.language = language

        from asrecognition import ASREngine
        language_models = {
            'es': "jonatasgrosman/wav2vec2-large-xlsr-53-spanish",
            'en': 'facebook/wav2vec2-base'
        }
        assert language in language_models, f'There is model currently associated with the language {language}. Plesae, modify the code'
        self.tr = ASREngine(self.language, model_path=language_models[self.language])

    # @decorator_timer
    def transcribe(self, file: Path):
        result = self.tr.transcribe([str(file)])
        return result[0]['transcription'].capitalize()


class WhisperTranscriber(Transcriber):
    def __init__(self, size: str = 'base'):
        """ whisper CLI & models for all languages
        --model
        Tiny
        Base
        Small
        Medium
        Large 2.87 gb
        --language 
        Spanish
        """
        # import whisper
        # model = whisper.load_model(model_name)
        import pywhisper
        self.tr = pywhisper.load_model(size)

    def transcribe(self, file: Path):
        """
        output sample from whisper:
        {'text': ' Este debería hacer el primer segmento.',
        'segments': [{'id': 0, 'seek': 0, 'start': 0.0, 'end': 3.0, 'text': ' Este debería hacer el primer segmento.', 'tokens': [50364, 16105, 29671, 2686, 6720, 806, 12595, 9469, 78, 13, 50514], 'temperature': 0.0, 'avg_logprob': -0.4960346221923828, 'compression_ratio': 0.8260869565217391, 'no_speech_prob': 0.02002035826444626}],
        'language': 'es'}
        :param file:
        :return:
        """
        result = self.tr.transcribe(str(file))
        return result["text"]


""" DEBUGGING
import audio_utils

file_path = Path('./OFFLINE/en.wav')
transcriber = Wave2vecTranscriber(language="en")
# transcriber = Wave2vecTranscriber(language="es")
# transcriber = WhisperTranscriber(size='tiny')
transcription = audio_utils.split_transcription(transcriber, file_path)
"""
