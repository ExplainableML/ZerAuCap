from glob import glob
import pandas as pd
import librosa
import os
import json


def get_length_summary_stats(file_list, dataset, json_=None):
    if dataset == "Clotho":
        audio_ts = [librosa.get_duration(path=file) for file in file_list]
        print("Clotho samples: {}".format(len(audio_ts)))
        sum_stats = pd.DataFrame([audio_ts]).transpose().describe()

    elif dataset == "AudioCaps":
        f = open(json_)
        sound_file_json = json.load(f)
        sound_file_suffixes = [file["sound_name"] for file in sound_file_json]
        print("AC samples: {}".format(len(sound_file_suffixes)))

        data = [
            librosa.get_duration(path=file)
            for file in file_list
            if os.path.split(file)[-1] in sound_file_suffixes
        ]
        sum_stats = pd.DataFrame([data]).transpose().describe()

    else:
        pass

    return sum_stats


if __name__ == "__main__":
    ac_val = get_length_summary_stats(
        file_list=glob("../softlinks/AudioCaps_data/*.wav"),
        dataset="AudioCaps",
        json_="AudioCaps/AudioCaps_val.json",
    )

    ac_test = get_length_summary_stats(
        file_list=glob("../softlinks/AudioCaps_data/*.wav"),
        dataset="AudioCaps",
        json_="AudioCaps/AudioCaps_test.json",
    )

    clotho_test = get_length_summary_stats(
        file_list=glob("Clotho/test_sounds/*.wav"), dataset="Clotho"
    )

    sum_stats = pd.concat([ac_val, ac_test, clotho_test], axis=1).round(1)

    sum_stats.columns = [
        "AudioCaps Validation Set",
        "AudioCaps Test Set",
        "Clotho Test Set",
    ]

    print(sum_stats.to_latex(caption="Summary statistics of the used datasets"))
