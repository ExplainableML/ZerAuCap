# %%
import openai
from tqdm import tqdm
from joblib import Parallel, delayed, parallel_backend
import pandas as pd
import time
import csv
import itertools


def get_chat_gpt_answers(
    prompt_1, prompt_2, answer_1, temperature, system_behavior_message=None
):
    if system_behavior_message != None:
        chat_gpt_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_behavior_message},
                {"role": "user", "content": prompt_1},
                {"role": "assistant", "content": answer_1},
                {"role": "user", "content": prompt_2},
            ],
            temperature=temperature,
        )

    return chat_gpt_response["choices"][0]["message"]["content"]


def clean_chat_gpt_response(response):
    start_of_list = response.find("[")
    end_of_list = response.rfind("]")
    all_variations = response[start_of_list : end_of_list + 1]

    return eval(all_variations)


def batch(keyword_list, batch_size):
    return [
        keyword_list[i : i + batch_size]
        for i in range(0, len(keyword_list), batch_size)
    ]


def get_additional_kws(keywords, m_variations_wanted, temperature):
    time.sleep(60)  # due to API limit

    response = get_chat_gpt_answers(
        prompt_1="Create "
        + str(m_variations_wanted)
        + " variations of each object in the following list of audio tags,\
                                while each variation is not longer than 4 words! list_of_audio_objects=[car, cat, people]",
        answer_1="variations=['electric car', 'breaking car', 'accelerating car', \
                                'meowing cat', 'screaming cat', 'purring cat', 'talking people', 'laughing people', 'crying people']",
        prompt_2="Do the same for every object in the list: "
        + str(keywords)
        + ". Append them to one Python list and only\
                                provide the list 'all_variations'! Do not skip any objects!",
        temperature=temperature,
        system_behavior_message="You are an assistant creating variations of audio tags that you are provided with. \
                                You only answer in Python lists! You do not answer in a polite way. You merely communicate using Python lists!\
                                Your answers do not contain any of the inputs!",
    )

    clean_response = clean_chat_gpt_response(response)

    return clean_response


# %%

if __name__ == "__main__":
    print("Extending keyword list using ChatGPT...")

    path_to_keywords = "data/AudioSet/class_labels_indices.csv"
    keywords = list(pd.read_csv(path_to_keywords)["display_name"])
    keywords = [tag.strip() for tag in keywords for tag in tag.split(",")]

    print("Successfully read in initial keyword list")

    temperature = 0
    m_variations_wanted = 3
    l_sounding_objects_per_batch = 5

    print(
        "{} variations wanted, {} sounding objects per batch and ChatGPT Temperature = {}".format(
            m_variations_wanted, l_sounding_objects_per_batch, temperature
        )
    )

    keywords_batched = batch(keywords, l_sounding_objects_per_batch)

    with parallel_backend("multiprocessing"):
        results = Parallel(n_jobs=3, verbose=1)(
            delayed(get_additional_kws)(batch, m_variations_wanted, temperature)
            for batch in tqdm(keywords_batched)
        )

    results = list(itertools.chain.from_iterable(results))

    save_path = "./data/sounding_objects/chatgpt_audio_tags.csv"

    with open(save_path, mode="w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        for string in results:
            csv_writer.writerow([string])

    print("Written to CSV in: {}".format(save_path))
