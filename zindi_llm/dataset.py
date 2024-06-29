import json

from datasets import Dataset

from zindi_llm import (
    EXTRA_DATA,
    TESTING_KNOWLEDGE,
    TESTING_PATH,
    TRAINING_PATH,
)


def create_dataset(data: dict):
    def patch_raw(raw: dict):
        for i in range(3, 6):
            raw[f"option {i}"] = raw.get(f"option {i}")
        return raw

    data_pashed = [patch_raw(raw) for raw in data.values()]
    data_pashed = Dataset.from_list(data_pashed)
    return data_pashed


def load_datasets(use_extra_dataset: bool = False, use_all_class: bool = False):
    loads = lambda path: json.load(open(path))
    train, val = None, None

    test = loads(TESTING_PATH)
    extra = loads(EXTRA_DATA)
    test = {question_id: extra[question_id] for question_id in test}
    test = create_dataset(test)

    if not use_extra_dataset:
        train = loads(TRAINING_PATH)
        train = create_dataset(train)
        train = train.train_test_split(test_size=0.1, stratify_by_column="category")
        return train["train"], train["test"], test

    val = loads(TESTING_PATH)
    train = loads(EXTRA_DATA)
    train = {key: value for key, value in train.items() if key not in val}

    if not use_all_class:
        train = {
            key: value
            for key, value in train.items()
            if value["category"] in TESTING_KNOWLEDGE
        }
    train = create_dataset(train)
    train = train.train_test_split(test_size=0.1, stratify_by_column="category")
    return train["train"], train["test"], test
