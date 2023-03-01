# -*- coding: utf-8 -*-
# ! /usr/bin/python3
# 11-2020
__author__ = 'Alejandro Rojo Gualix'
"""
pip install pydub asrecognition pywhisper

https://github.com/openai/whisper
pywhisper = openai/whisper + extra features
https://github.com/fcakyon/pywhisper
"""

import sys
import argparse
from pathlib import Path

# PROJECT IMPORTS
import audio_utils
import model_classes


def get_output_file_name(file_path: Path, use_folder=False):
    if use_folder:
        folder = Path(file_path.parent, 'transcriptions')
        folder.mkdir(exist_ok=True)
        return Path(folder, file_path.stem + '.txt')
    else:
        return Path(file_path.parent, file_path.stem + '_transcription.txt')


# extensions = ['*.wav', '*.ogg', '*.mp3']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""Transcripts audio file or audios folder""")
    # positional
    parser.add_argument('input', metavar='input file', type=str,
                        help='input path (file or folder)')
    parser.add_argument('-r', '--recursive', action='store_true',
                        help='Use it along a folder input: it makes available all files within subfolders')
    parser.add_argument('-f', '--folder',
                        help='Output transcription text to folder transcriptions')
    # optional & mutually exclusive
    parser.add_argument("-m", "--model", type=str, default='whisper', choices=["whisper", "wave2vec2"],
                        help="Choose model for transcription")

    args = parser.parse_args()
    print(args)

    input_path = Path(args.input)
    assert input_path.exists(), 'input file/folder not found'

    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        # get input files
        if args.recursive:
            files = input_path.rglob('*.wav')
        else:
            files = input_path.glob('*.wav')
        assert files, 'No compatible files found in ' + str(path2.resolve())

    if args.model == 'whisper':
        transcriber = model_classes.WhisperTranscriber(size='tiny')  # 13s audio -> 10.5 (including preprocessing)
        # transcriber = model_classes.WhisperTranscriber(size='base')  # 13s audio -> 19.3 (including preprocessing)
        # transcriber = model_classes.WhisperTranscriber(size='large')  # 13s audio -> 330 (including preprocessing)
    elif args.model == 'wave2vec2':
        # transcriber = model_classes.Wave2vecTranscriber(language="es")
        transcriber = model_classes.Wave2vecTranscriber(language="en")

    for file_path in files:
        print('##', file_path.name, '##')
        transcription = audio_utils.split_transcription(transcriber, file_path)
        output_file_name = get_output_file_name(file_path)
        # FIXME
        # output_file_name = get_output_file_name(file_path, use_folder=args.folder)
        with output_file_name.open('w') as f:
            f.write(transcription)
        print('##', 'saved to', output_file_name, '##')
