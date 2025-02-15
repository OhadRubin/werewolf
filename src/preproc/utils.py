import io
from pydub import AudioSegment
import copy


def utterence_to_audio_snippet(start_time_in_seconds, end_time_in_seconds, audio):
    start_ms = start_time_in_seconds * 1000
    end_ms = end_time_in_seconds * 1000

    # Extract the audio segment
    snippet = audio[start_ms:end_ms].set_frame_rate(24000).set_channels(1)
    audio_buffer = io.BytesIO()
    snippet.export(audio_buffer, format="wav")
    return audio_buffer.getvalue()


def construct_audio_segment_and_transcript(dialogues, idx, K=10):
    curr = dialogues[idx]
    start, end = curr["start_time_in_seconds"], curr["end_time_in_seconds"]
    utterances = [curr]

    if start == 0 or idx == 0:
        return start, end, utterances

    # Add the previous utterances that start within K second window
    prev_idx = idx - 1
    while prev_idx >= 0:
        prev = dialogues[prev_idx]
        if prev["start_time_in_seconds"] < min(start - K, 0):
            break
        utterances = [prev] + utterances
        prev_idx -= 1

    start = max(0, start - K)  # Start 10 seconds before current utterance
    return start, end, copy.deepcopy(utterances)


def construct_spans(datum, K=10):
    dialogues = datum["Dialogue"]
    span_list = []
    for i in range(len(dialogues)):
        start, end, utterances = construct_audio_segment_and_transcript(
            dialogues, i, K=K
        )
        span_list.append((start, end, utterances))
    return span_list


def dialog_to_audio_list(datum, folder):
    audio_file = datum["file_name"]
    audio = AudioSegment.from_file(f"{folder}/{audio_file}")
    dialogue_data = []
    for start, end, utterances in construct_spans(datum):
        audio_data = utterence_to_audio_snippet(start, end, audio)
        dialogue_data.append((audio_data, utterances))

    return dialogue_data
