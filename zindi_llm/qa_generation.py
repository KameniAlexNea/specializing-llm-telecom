import gc
import os
import random
import uuid

import torch
from docx import Document
from peft import PeftModelForCausalLM
from tqdm import tqdm
from transformers import AutoTokenizer
from unsloth import FastLanguageModel

max_new_tokens = 2048
max_seq_length = 4096  # Choose any! We auto support RoPE Scaling internally!
dtype = (
    None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
)
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.
NUM_QUESTIONS = 5

MODEL_PATH = (
    "unsloth/llama-3-8b-Instruct-bnb-4bit"  # "unsloth/llama-3-8b-Instruct-bnb-4bit"
)

QUERY = """
Please generate <num_questions>{NUM_QUESTIONS}</num_questions> questions based on the provided context.

Read the context carefully and create questions that can be clearly answered using only the information provided. Do not make questions that would require outside knowledge to answer.

For each question you create:
- Write out the question text inside "question" quotes 
- Generate 5 possible answer options and list them out as "option 1", "option 2", etc. Make sure the incorrect options are plausible based on the context.
- Specify the correct answer inside "answer" quotes, referring to the option number

Output each complete question with its answer options and correct answer inside <qa> tags, using this format:

<qa>
"question": "Question text goes here",
"option 1": "First answer option",
"option 2": "Second answer option", 
"option 3": "Third answer option",
"option 4": "Fourth answer option",
"option 5": "Fifth answer option",
"answer": "option X: Correct answer text",
</qa>

Generate all the questions one after another until you have created the requested number based on the context provided. 
Do not repeat questions. If possible, avoid using "What is the purpose" while formulating question. 
Context has been randomly extracted from document so, number at the beginning of the text may be related to section number, and directly related to the context

Here is the context to generate questions from:

<context>
{CONTEXT}
</context>
"""


def save_text(text: str, base_folder: str = "data/generated_qa/"):
    name = str(uuid.uuid1()) + ".txt"
    with open(os.path.join(base_folder, name), "w") as f:
        f.write(text)


def load_model():
    result: tuple[PeftModelForCausalLM, AutoTokenizer] = (
        FastLanguageModel.from_pretrained(
            model_name=MODEL_PATH,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
        )
    )
    model: PeftModelForCausalLM = result[0]
    tokenizer: AutoTokenizer = result[1]

    FastLanguageModel.for_inference(model)
    return model, tokenizer


def prepare_prompt(text, tokenizer: AutoTokenizer):
    chat = [
        {"role": "user", "content": text},
    ]
    prompt = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )
    return prompt


@torch.no_grad()
def make_prediction(
    model: PeftModelForCausalLM, tokenizer: AutoTokenizer, texts: list[str]
) -> list[str]:
    texts = [
        prepare_prompt(
            QUERY.format(NUM_QUESTIONS=NUM_QUESTIONS, CONTEXT=text), tokenizer
        )
        for text in texts
    ]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(
        "cuda"
    )
    predicted = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
    )
    return tokenizer.batch_decode(predicted)


def load_pdf_data(path: str, reject: float = 0.2, take_n=2048, n_sentence=250):
    document = Document(path)
    texts = [par.text for par in document.paragraphs]
    texts = texts[int(len(texts) * reject) : -int(len(texts) * reject / 3)]
    take_n = min(int((len(texts) - n_sentence) * 0.75), take_n)

    index = random.sample(range(len(texts) - n_sentence), take_n)
    texts = ["\n".join(texts[i : i + n_sentence]) for i in index]
    return texts


def make_document_prediction(
    path: str,
    batch_size: int = 8,
    file_name: str = "document",
    reject: float = 0.2,
    take_n=2048,
    n_sentence=250,
):
    folder = os.path.join("data/generated_qa/", file_name)
    os.makedirs(folder, exist_ok=True)

    texts = load_pdf_data(path, reject, take_n, n_sentence)
    model, tokenizer = load_model()
    for i in tqdm(range(0, len(texts), batch_size)):
        text = texts[i : i + batch_size]
        llm_answers = make_prediction(model, tokenizer, text)
        list(map(lambda x: save_text(x, folder), llm_answers))
        gc.collect()
