import random

import pandas as pd
from datasets import Dataset

alpaca_prompt = """You are an AI assistant specialized in the 3rd Generation Partnership Project (3GPP) Release specifications. The Standards Development: 3GPP is known for creating standards for 3G (third generation), 4G (fourth generation), and 5G (fifth generation) mobile networks. 
Your task is to analyze questions and select the most appropriate answer from given options based on your knowledge of 3GPP standards.

<instruction>
{}
</instruction>

<options>
{}
</options>

<reponse>
{}
</reponse>"""


def prepare_text(examples: dict[str, str]):
    if "question" not in examples:
        return examples

    def apply_one(question, answer, *options):
        options = [i for i in options if i is not None]
        index = list(range(len(options)))
        random.shuffle(index)
        option_text = "\n".join(
            f"option {j+1}: " + options[i] for j, i in enumerate(index)
        )
        response = f"option {index.index(answer - 1) + 1}: " + options[answer - 1]
        return alpaca_prompt.format(str(question), option_text, response)

    if isinstance(examples["question"], str):
        return apply_one(
            examples["question"],
            examples["answer_index"],
            examples["option 1"],
            examples["option 2"],
            examples["option 3"],
            examples["option 4"],
            examples["option 5"],
        )
    texts = [
        apply_one(question, answer, *options)
        for question, answer, *options in zip(
            examples["question"],
            examples["answer_index"],
            examples["option 1"],
            examples["option 2"],
            examples["option 3"],
            examples["option 4"],
            examples["option 5"],
        )
    ]
    return texts


def load_generated_dataset():
    FOLDER = "data/full_data/"
    generated = pd.read_csv(FOLDER + "generated_v1.csv", keep_default_na=False)
    train = pd.read_csv(FOLDER + "train_data.csv", keep_default_na=False)
    data = pd.concat([train, generated], ignore_index=True)

    test_df = pd.read_csv(FOLDER + "test_data.csv", keep_default_na=False)

    train = Dataset.from_pandas(data)  # .with_transform(prepare_text)
    test = Dataset.from_pandas(test_df)  # .with_transform(prepare_text)

    train = train.train_test_split(test_size=0.05, seed=2333)  # try random generation
    return train["train"], train["test"], test
