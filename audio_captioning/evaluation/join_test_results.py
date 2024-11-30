from glob import glob
import os
import pandas as pd
import numpy as np
import argparse
import sys


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_files_path",
        type=str,
        help="the folder where the result file csv files are stored in",
    )
    parser.add_argument(
        "--hyperparam_json_path",
        type=str,
        help="specify path to json that stores all hyperparams",
    )
    parser.add_argument("--caption", type=str, help="specify caption for table")
    parser.add_argument(
        "--SOTA_table", type=str, help="specify TRUE if table is SOTA table"
    )
    return parser.parse_args()


def sorting_function_keyword_lists(list_name):
    return len(str(list_name))


if __name__ == "__main__":
    args = parse_config()

    # hyperparams

    hyperparams_dict = glob(
        os.path.join(args.hyperparam_json_path, "**/*.json"), recursive=True
    )[0]
    hyperparams = pd.read_json(hyperparams_dict).loc[
        0:1,
        [
            "alpha",
            "beta",
            "k",
            "temperature",
            "l",
            "keyword_prompt",
            "prompt",
            "end_penalty",
        ],
    ]

    latex_hyperparams = hyperparams.to_latex(index=False, header=True)
    # print(latex_hyperparams)

    result_files = []

    for f in glob(os.path.join(args.result_files_path, "**/*.csv"), recursive=True):
        if ("2023-06" in f) and ("test_performance" in f):
            result_files.append(pd.read_csv(f).set_index(["Dataset", "Model"]))

    result_table = (
        pd.concat(result_files, axis=0)
        .applymap(lambda x: x * 100)
        .rename(columns={"Mean_NLG_M": "NLG Mean Score"})
        .sort_values("NLG Mean Score")
    )

    baselines = result_table.reset_index()

    baselines = baselines[baselines["Model"] == "Baseline_LM_only"].set_index(
        ["Dataset", "Model"]
    )

    if args.SOTA_table == "TRUE":
        result_table.reset_index(inplace=True)
        result_table = result_table.loc[
            result_table.groupby("Dataset")["SPIDEr"].idxmax()
        ].set_index(["Dataset", "Model"])
        result_table = pd.concat([result_table, baselines], axis=0)

    # add supervised SOTA results for clotho and AudioCaps

    SOTA = pd.DataFrame(
        {
            "Dataset": ["AudioCaps", "clotho"],
            "Model": ["Supervised SOTA", "Supervised SOTA"],
            "Bleu_1": [70.7, 60.1],
            "Bleu_2": [53.4, 40.0],
            "Bleu_3": [39.5, 27.1],
            "Bleu_4": [28.3, 18.2],
            "METEOR": [25.0, 18.5],
            "ROUGE_L": [50.7, 40.0],
            "CIDEr": [78.7, 48.8],
            "SPICE": [18.2, 13.5],
            "SPIDEr": [48.5, 31.0],
        }
    ).set_index(["Dataset", "Model"])

    ablation_table = pd.concat([result_table, SOTA], axis=0).reset_index()

    groups = [
        ablation_table["Model"].str.contains("Baseline"),
        ablation_table["Model"].str.contains("AudioCLIP"),
        ablation_table["Model"].str.contains("LAION"),
        ablation_table["Model"].str.contains("MAGIC_WavCaps"),
        ablation_table["Model"].str.contains("WavCaps"),
        ablation_table["Model"].str.contains("SOTA"),
    ]

    group_names = [
        "Baseline",
        "MAGIC AudioCLIP",
        "MAGIC LAION",
        "MAGIC WavCaps",
        "WavCaps",
        "SOTA",
    ]

    print(ablation_table)
    ablation_table["Group"] = pd.Series(np.select(groups, group_names))
    ablation_table["Dataset"] = np.where(
        ablation_table["Dataset"] == "clotho", "Clotho", ablation_table["Dataset"]
    )

    ablation_table = ablation_table.groupby(["Dataset", "Group"]).apply(
        lambda x: x.sort_values("Bleu_1")
    )

    ablation_table = ablation_table.reset_index(level=[0, 1, 2], drop=True).drop(
        columns=["Group", "Dataset"]
    )

    ablation_table["MAGIC"] = np.where(
        ablation_table["Model"].str.contains("MAGIC"), "On", "Off"
    )

    audio_models = [
        ablation_table["Model"].str.contains("WavCaps"),
        ablation_table["Model"].str.contains("LAION"),
        ablation_table["Model"].str.contains("AudioCLIP"),
        ablation_table["Model"].str.contains("Baseline"),
        ablation_table["Model"].str.contains("SOTA"),
    ]

    audio_model_names = ["WavCaps", "LAION", "AudioCLIP", "-", "Supervised SOTA"]

    ablation_table["Audio Model"] = np.select(audio_models, audio_model_names)

    keywords = [
        ablation_table["Model"].str.contains("ChatGPT"),
        (ablation_table["Model"].str.contains("KW"))
        & (~ablation_table["Model"].str.contains("ChatGPT")),
        ~ablation_table["Model"].str.contains("KW"),
    ]

    keyword_labels = ["AudioSet+ChatGPT KW", "AudioSet KW", "-"]
    ablation_table["Keywords"] = np.select(keywords, keyword_labels)

    name_cols = ["MAGIC", "Audio Model", "Keywords"]
    new_columns = name_cols + (ablation_table.columns.drop(name_cols).tolist())
    ablation_table = ablation_table[new_columns].drop(
        columns=["Model", "Bleu_2", "Bleu_3"]
    )

    ablation_table["MAGIC"] = np.where(
        (ablation_table["Audio Model"] == "-") & (ablation_table["Keywords"] == "-"),
        "Off",
        ablation_table["MAGIC"],
    )

    if args.SOTA_table != "TRUE":
        ablation_table = ablation_table[
            ~ablation_table["Audio Model"].str.contains("SOTA")
        ]

    else:
        # include SOTA, Baseline and our best model

        ablation_table = ablation_table.loc[
            ablation_table["Audio Model"].str.contains("SOTA")
            | ablation_table["Audio Model"].str.contains("-")
            | (
                ablation_table["Audio Model"].str.contains("WavCaps")
                & ablation_table["Keywords"].str.contains("AudioSet KW")
                & ablation_table["MAGIC"].str.contains("On")
            )
        ]

    index = [""] * ablation_table.shape[0]
    index[0] = "AudioCaps"
    index[len(index) // 2] = "Clotho"
    ablation_table.index = index
    ablation_table.index.name = "Dataset"

    ablation_table.columns = ablation_table.columns.str.replace("_", " ")
    # ablation_table["Model"] = ablation_table["Model"].str.replace('_', ' ')

    # Create one ablation table for every dataset

    ac = ablation_table.iloc[: len(ablation_table) // 2]
    clotho = ablation_table.iloc[len(ablation_table) // 2 :]

    ac = (
        ac.groupby(["MAGIC", "Audio Model", "Keywords"])
        .apply(
            lambda x: x.assign(
                keyword_list_length=x["Keywords"].map(sorting_function_keyword_lists)
            ).sort_values("Bleu 1")
        )
        .reset_index(drop=True)
        .drop(columns=["keyword_list_length"])
    )

    clotho = (
        clotho.groupby(["MAGIC", "Audio Model", "Keywords"])
        .apply(
            lambda x: x.assign(
                keyword_list_length=x["Keywords"].map(sorting_function_keyword_lists)
            ).sort_values("SPIDEr")
        )
        .reset_index(drop=True)
        .drop(columns=["keyword_list_length"])
    )

    ablation_tables = {"AudioCaps": ac, "Clotho": clotho}

    for name, table in ablation_tables.items():
        if "AudioCaps" in name:
            caption = args.caption + " on AudioCaps"

        if "Clotho" in name:
            caption = args.caption + " on Clotho"

        # table.drop(columns=["Bleu 2", "Bleu 3"], inplace=True)

        latex_table = table.to_latex(index=False, caption=caption)
        print(latex_table)
        continue
        import sys

        sys.exit()

        if args.SOTA_table == "TRUE":
            path_name = "../evaluation/tables/" + "SOTA_table_" + name + ".txt"

        else:
            path_name = "../evaluation/tables/" + "ablation_table_" + name + ".txt"

        print("Writing table to {}".format(path_name))

        with open(path_name, "w") as f:
            f.write(latex_table)

#    latex_results = ablation_table.to_latex()

#   with open('ablation_table.txt', 'w') as f:
#      f.write(latex_results)
# take json and automatically include hyperparams in caption
