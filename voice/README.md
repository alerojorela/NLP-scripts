## Speech to text

### Requirements

```
pip install pydub asrecognition pywhisper
```

### Usage

Use Whisper to transcribe `en.wav`,  it will output the transcription into a file called `en_transcription.txt`:

```bash
python3 run.py OFFLINE/en.wav
```

Similarly, use Whisper to transcribe `en.wav` with Wave2vec2:

```bash
python3 run.py OFFLINE/en.wav -m wave2vec2
```

Use Whisper to transcribe all .wav files inside the folder (recursively) and output the transcription into text files with  `_transcription` suffix located in the same audio folder:

```bash
python3 run.py OFFLINE/ -r
```

