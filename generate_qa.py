import gc
import os

from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from zindi_llm.qa_generation import (
    load_model,
    load_pdf_data,
    make_prediction,
    save_text,
)

SAVE_FOLDER = "data/generated_qa/QCM_Context"
model, tokenizer = load_model()


def make_document_prediction(
    path: str,
    batch_size: int = 8,
    file_name: str = "document",
    reject: float = 0.2,
    take_n=2048,
    n_sentence=250,
):
    folder = os.path.join(SAVE_FOLDER, file_name)
    os.makedirs(folder, exist_ok=True)

    texts = load_pdf_data(path, reject, take_n, n_sentence)
    for i in tqdm(range(0, len(texts), batch_size)):
        text = texts[i : i + batch_size]
        llm_answers = make_prediction(model, tokenizer, text)
        list(map(lambda x: save_text(x, folder), llm_answers))
        gc.collect()


files = [
    "data/zindi_data/rel18/rel_17.docx"
]  # sorted(glob("data/zindi_data/rel18/*i*.docx"))

for file in tqdm(files[:]):
    name = os.path.basename(file).replace(".docx", "")
    make_document_prediction(
        file, file_name=name, batch_size=8, n_sentence=75, take_n=32
    )
    gc.collect()
