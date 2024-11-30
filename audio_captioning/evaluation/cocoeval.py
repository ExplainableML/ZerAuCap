from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import json
import os
import glob
from json import encoder
from datetime import datetime

encoder.FLOAT_REPR = lambda o: format(o, ".3f")

import argparse


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_file_path",
        type=str,
        help="the folder where the result file paths are stored in",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        help="file prefix for the json file that contains all metrics",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_config()

    result_files = []

    for f in glob.glob(
        os.path.join(args.result_file_path, "**/*.json"), recursive=True
    ):
        if "hyperparam_experiments" in f:
            result_files.append(f)
            # print(f)

    result_dicts = []

    for result_file in result_files:
        # calculate metrics and store
        cocoEval = COCOEvalCap(result_file)
        cocoEval.evaluate()
        # add to cocoEval.eval dict all hyperparams (alpha, beta, prompt, ...). include them in the dict in args.result_file_path

        res_dict = cocoEval.eval

        all_res_dict = json.load(open(result_file))

        # all_res_dict: list of dicts

        keys_to_remove = ["captions", "sound_name", "split", "prediction"]
        for k in keys_to_remove:
            del all_res_dict[0][k]

        if "gpt2" in result_file:
            all_res_dict[0]["model"] = "gpt2"

        elif "opt-1.3b" in result_file:
            all_res_dict[0]["model"] = "opt-1.3b"

        else:
            pass

        for hyperparam in all_res_dict[0].keys():
            res_dict[str(hyperparam)] = all_res_dict[0][str(hyperparam)]

        result_dicts.append(res_dict)

    filename = (
        str(args.prefix) + datetime.today().strftime("%Y_%m_%d_%H_%M_%S") + ".json"
    )
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(result_dicts, f, ensure_ascii=False, indent=4)

    # for metric, score in cocoEval.eval.items():
    #   print ('%s: %.3f'%(metric, score))
