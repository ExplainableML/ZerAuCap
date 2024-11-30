import pandas as pd


def create_dicts(in_f, split_name, filepath):
    """
    each item in the split has the following format
        {
            'split': train, val, or test
            'sound_name' xxx.wav,
            'file_path': ...,
            'captions': [sentence_1,
                        sentence_2,
                        ...,
                        sentence_5]
        }

    specify split_name to say if we read in train, val or test captions
    specify filepath?
    """
    all_items = []

    path = os.path.join(in_f, split_name + ".csv")

    data = pd.read_csv(path, sep=",")

    for _, item in data.iterrows():
        information = {
            "split": split_name,
            "sound_name": item.file_name,
            "file_path": filepath,
            "captions": list(item[1:6]),
        }

        all_items.append(information)

    return all_items


if __name__ == "__main__":
    import os

    save_path = r"./clotho_v2.1/"  # save prepared data here
    if os.path.exists(save_path):
        pass
    else:  # recursively construct directory
        os.makedirs(save_path, exist_ok=True)

    import json

    in_f = r"./raw_data/caption_datasets/clotho_v2.1"

    splits = {
        split_name: create_dicts(in_f, split_name=split_name, filepath="WHAT_IS_THIS?")
        for split_name in ["train", "val", "test"]
    }

    print(
        "Number of train instance {}, val instances {}, and test instances {}".format(
            len(splits["train"]), len(splits["val"]), len(splits["test"])
        )
    )

    train_save_path = save_path + "/" + r"clotho_v2.1_train.json"
    with open(train_save_path, "w") as outfile:
        json.dump(splits["train"], outfile, indent=4)

    val_save_path = save_path + "/" + r"clotho_v2.1_val.json"
    with open(val_save_path, "w") as outfile:
        json.dump(splits["val"], outfile, indent=4)

    test_save_path = save_path + "/" + r"clotho_v2.1_test.json"
    with open(test_save_path, "w") as outfile:
        json.dump(splits["test"], outfile, indent=4)
