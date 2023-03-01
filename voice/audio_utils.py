from time import time
from pathlib import Path

from pydub import AudioSegment
from pydub.silence import split_on_silence

# Create a silence chunk that's 0.5 seconds (or 500 ms) long for padding.
silence_padding = AudioSegment.silent(duration=500)

# PROJECT IMPORTS


def decorator_timer(some_function):
    def wrapper(*args, **kwargs):
        start_time = time()

        result = some_function(*args, **kwargs)

        execution_time = time() - start_time
        print(f'Processing lasted {str(execution_time)} seconds\n')

        return result

    return wrapper


# region "Audio utilities"
# Define a function to normalization a chunk to a target amplitude.
def match_target_amplitude(aChunk, target_dBFS):
    ''' Normalize given audio chunk '''
    change_in_dBFS = target_dBFS - aChunk.dBFS
    return aChunk.apply_gain(change_in_dBFS)


def split_by_silence(file_path: Path,
                     chunk_min_length=2000,
                     silence_relative_db: int = 16, min_silence_len=1000):
    # https://stackoverflow.com/questions/45526996/split-audio-files-using-silence-detection
    '''
    min_silence_len
    target_length
    :param file_path:
    :param silence_relative_db:
    :return:
    '''
    """
    WARNING:root:Some files (['../data/llamadas chatbot/converted16kHz/B2.ogg']) have more than 120 seconds.
    To prevent memory issues we highly recommend you to split them into smaller chunks of less than 120 seconds.
    """
    sound = AudioSegment.from_wav(file_path)
    dBFS = sound.dBFS
    # Consider a chunk silent if it's quieter than -silence_relative_db dBFS.
    chunks = split_on_silence(sound,
                              min_silence_len=min_silence_len,
                              silence_thresh=dBFS - silence_relative_db,
                              keep_silence=250  # optional
                              )

    # Merge small segments
    output_chunks = [chunks[0]]
    for chunk in chunks[1:]:
        if len(output_chunks[-1]) < chunk_min_length:
            output_chunks[-1] += silence_padding + chunk  # append
        else:  # if the last output chunk is longer we create a new one
            output_chunks.append(chunk)
    return output_chunks


@decorator_timer
def split_transcription(transcriber, file_path: Path, split_on_silence_kwargs={}):
    """
    transcripción de un archivo de sonido segmentándolo por silencios para mejorar el uso de la memoria
    :param transcriber:
    :param file_path:
    :param split_on_silence_kwargs:
    :return:
    """

    temp_folder_path = Path(file_path.parent, 'temp')
    temp_folder_path.mkdir(exist_ok=True)

    chunks = split_by_silence(file_path.absolute(), **split_on_silence_kwargs)

    output = []
    # Process each chunk with your parameters
    for i, chunk in enumerate(chunks):
        # Normalize the entire chunk.
        normalized_chunk = match_target_amplitude(chunk, -20.0)
        # Add the padding chunk to beginning and end of the entire chunk.
        normalized_chunk = silence_padding + normalized_chunk + silence_padding

        # Export the audio chunk with new bitrate.
        output_file = Path(temp_folder_path, f"chunk{i}.mp3")
        normalized_chunk.export(str(output_file.absolute()), bitrate="192k", format="wav")
        # print(f"Exporting chunk{i}".format(i))

        transcription = transcriber.transcribe(str(output_file.absolute()))

        output.append(transcription)
        print(transcription)

    return '\n'.join(output)

# endregion
