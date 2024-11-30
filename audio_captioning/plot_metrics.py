import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# Replace 'your_path' with the path of the webpage containing the HTML table
path = "inference_result/gpt2/AudioCaps/excludes_prompt_magic/output_tables/test_performance/0.119_2023-09-29 21:46:23_MAGIC_WavCaps_AudioSet_KW.html"

# Read the HTML table from the path into a list of DataFrames
tables_list = pd.read_html(path)

# Assuming the table of interest is the first table on the webpage
# If the table you want is not the first one, specify the index accordingly
desired_table_index = 0
df = tables_list[desired_table_index][1:]
print(df.columns)

df["sentence_length"] = df["pred"].apply(lambda x: len(x.split(" ")))

# iterate over metrics
for metric in [
    "Bleu_1",
    "Bleu_2",
    "Bleu_3",
    "Bleu_4",
    "METEOR",
    "ROUGE_L",
    "CIDEr",
    "SPICE",
    "SPIDEr",
]:
    # plot
    sns.jointplot(data=df, x="sentence_length", y=metric, kind="reg")
    plt.savefig(f"{metric}_vs_sentence_length.png")
# sns.jointplot(data=df, x="sentence_length", y="CIDEr", kind="reg")
# plt.savefig("bleu4_vs_sentence_length.png")
# Display the DataFrame
# print(df)
