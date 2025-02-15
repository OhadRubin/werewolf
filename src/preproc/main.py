import nltk
import json
import glob
import re
from moviepy.editor import VideoFileClip
import json
import os
import copy
import nltk

nltk.download("punkt_tab")

def sanitize(x):
    for punc in ["：", "，", "｜", "  ", " ", "#"]:
        x = x.replace(punc, " ").strip()
    x = x.replace("  ", " ").strip()
    x = re.sub(r"[^a-zA-Z0-9\s]", "", x)
    return x.strip()


def align_file_names_and_video_names(dataset_dir, downloaded_glob):
    with open(f"{dataset_dir}/youtube_urls_released.json", "r") as f:
        youtube_urls = json.load(f)
    existing_files = glob.glob(downloaded_glob)
    urls = list(youtube_urls.values())
    missing_urls = []
    for url in urls:
        if any(url in existing_file for existing_file in existing_files):
            continue
        missing_urls.append(url)

    video_names = []
    for split in ["train", "val", "test"]:
        data_file = f"{dataset_dir}/split/{split}.json"
        with open(data_file, "r") as f:
            data = json.load(f)
        video_names.extend(list({datum["video_name"] for datum in data}))

    video_names = list(set(video_names))
    file_names = [
        (sanitize(x.split("/")[-1].split("[")[0].strip()), x) for x in existing_files
    ]
    video_names = [(sanitize(x), x) for x in video_names]
    file_names = sorted(file_names, key=lambda x: x[0])
    video_names = sorted(video_names, key=lambda x: x[0])
    mapping = {}
    for game_id, ((f, old_f), (g, old_g)) in enumerate(zip(file_names, video_names)):
        assert f == g, f"{f} != {g}"
        mapping[old_g] = (game_id, old_f)
    return mapping


def timestamp_to_seconds(timestamp):
    h, m, s = map(int, timestamp.split(":"))
    return h * 3600 + m * 60 + s


# dataset_dir="/Users/ohadr/Werewolf/youtube",
# downloaded_glob="/Volumes/4TB/werewolf/raw_files/*",
# output_folder = "/Volumes/4TB/werewolf/saved_games"
def splice_audio_from_video_and_save_to_file(dataset_dir, downloaded_glob, output_folder):
    mapping = align_file_names_and_video_names(
        dataset_dir=dataset_dir,
        downloaded_glob=downloaded_glob,
    )
    for split in ["train", "val", "test"]:
        with open(f"{dataset_dir}/split/{split}.json", "r") as f:
            games = json.load(f)
            for datum in games:
                file_idx, video_name = mapping[datum["video_name"]]
                output_filename = f"{output_folder}/file_{file_idx}_{datum['Game_ID']}_clip.mp3"  # Using Game_ID from metadata
                if os.path.exists(output_filename):
                    continue

                start_seconds = timestamp_to_seconds(datum["startTime"])
                end_seconds = timestamp_to_seconds(datum["endTime"])

                # Extract audio segment
                with VideoFileClip(video_name) as video:
                    audio = video.audio.subclip(start_seconds, end_seconds)
                    # Save audio segment
                    audio.write_audiofile(output_filename)

    return mapping


def get_player_start_roles(datum):
    players = datum["playerNames"]
    start_roles = datum["startRoles"]
    return dict(zip(players, start_roles))


def get_player_end_roles(datum):
    players = datum["playerNames"]
    end_roles = datum["endRoles"]
    return dict(zip(players, end_roles))


def get_duration_seconds(start_time, end_time):
    start_parts = start_time.split(":")
    end_parts = end_time.split(":")

    if len(start_parts) == 3:
        start_hr, start_min, start_sec = start_parts
        start_total = int(start_hr) * 3600 + int(start_min) * 60 + int(start_sec)
    else:
        start_min, start_sec = start_parts
        start_total = int(start_min) * 60 + int(start_sec)

    if len(end_parts) == 3:
        end_hr, end_min, end_sec = end_parts
        end_total = int(end_hr) * 3600 + int(end_min) * 60 + int(end_sec)
    else:
        end_min, end_sec = end_parts
        end_total = int(end_min) * 60 + int(end_sec)

    return end_total - start_total


def parse_timestamp(timestamp):
    min, sec = timestamp.split(":")
    return int(min) * 60 + int(sec)


def add_endtimes(datum):
    dialogue = datum["Dialogue"]
    new_dialogue = []
    final_time = get_duration_seconds(datum["startTime"], datum["endTime"])
    start_roles = get_player_start_roles(datum)
    end_roles = get_player_end_roles(datum)

    for i in range(len(dialogue) - 1):
        curr = copy.deepcopy(dialogue[i])
        curr["start_time_in_seconds"] = parse_timestamp(dialogue[i]["timestamp"])
        next_time = parse_timestamp(dialogue[i + 1]["timestamp"])
        curr["end_time_in_seconds"] = min(next_time + 3, final_time)
        curr.pop("Rec_Id")
        curr.pop("timestamp")
        curr["target"] = ", ".join(curr["annotation"])
        curr.pop("annotation")
        curr["start_role"] = start_roles.get(curr["speaker"], "unknown")
        curr["end_role"] = end_roles.get(curr["speaker"], "unknown")

        new_dialogue.append(curr)

    # Handle last utterance
    last = copy.deepcopy(dialogue[-1])
    last["start_time_in_seconds"] = parse_timestamp(last["timestamp"])
    last["end_time_in_seconds"] = final_time
    new_dialogue.append(last)

    return new_dialogue


def add_endtimes_to_dataset(dataset_dir, downloaded_glob, audio_output_folder, text_output_folder):
    mapping = splice_audio_from_video_and_save_to_file(
        dataset_dir, downloaded_glob, audio_output_folder
    )

    final_data = {}
    for split in ["train", "val", "test"]:
        with open(f"{dataset_dir}/split/{split}.json", "r") as f:
            data = json.load(f)
        for datum in data:
            datum["Dialogue"] = add_endtimes(datum)
            file_idx, video_name = mapping[datum["video_name"]]
            datum["file_name"] = f"file_{file_idx}_{datum['Game_ID']}_clip.mp3"
        final_data[split] = data
        os.makedirs(text_output_folder, exist_ok=True)
        with open(f"{text_output_folder}/final_{split}.json", "w") as f:
            json.dump(final_data[split], f, indent=4)


    # Save mapping to JSON file
    with open(f"{audio_output_folder}/mapping.json", "w") as f:
        json.dump(mapping, f, indent=4)


import fire


# dataset_dir="/Users/ohadr/Werewolf/youtube",
# downloaded_glob="/Volumes/4TB/werewolf/raw_files/*",
# output_folder = "/Volumes/4TB/werewolf/saved_games"
# usage: python main.py add_endtimes_to_dataset  --dataset_dir /Users/ohadr/Werewolf/youtube --downloaded_glob "/Volumes/4TB/werewolf/raw_files/*" --audio_output_folder /Volumes/4TB/werewolf/saved_games --text_output_folder /Users/ohadr/Werewolf/youtube/split 
if __name__ == "__main__":
    fire.Fire(add_endtimes_to_dataset)
