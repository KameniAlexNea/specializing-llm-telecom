import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

os.environ["WANDB_PROJECT"] = "qa_telcom"
os.environ["WANDB_LOG_MODEL"] = "true"
os.environ["WANDB_WATCH"] = "none"
os.environ["WANDB_NOTEBOOK_NAME"] = "qa_telcom"

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

from zindi_llm.dataset import load_datasets


max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
dtype = (
    None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
)
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

model_name = "unsloth/Phi-3-mini-4k-instruct"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)


model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)


print(model)

train_ds, val_ds, test_ds = load_datasets(True, True)
print(len(train_ds), len(val_ds), len(test_ds))


print(train_ds[:5])


alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

OPTIONS = [f"option {i}" for i in range(1, 6)]
EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN


def formatting_prompts_func(examples: dict[str, str]):
    def apply_one(question, answer, category, *options):
        instructions = f"Domain: {category}: {question}"
        inputs = "\n".join(
            [
                (f"option {i}: " + text)
                for i, text in enumerate(options, start=1)
                if text is not None
            ]
        )
        outputs = answer
        return alpaca_prompt.format(instructions, inputs, outputs)

    texts = [
        apply_one(question, answer, category, *options)
        for question, answer, category, *options in zip(
            examples["question"],
            examples["answer"],
            examples["category"],
            examples["option 1"],
            examples["option 2"],
            examples["option 3"],
            examples["option 4"],
            examples["option 5"],
        )
    ]
    return {
        "text": texts,
    }


train_ds = train_ds.map(formatting_prompts_func, batched=True)
val_ds = val_ds.map(formatting_prompts_func, batched=True)
test_ds = test_ds.map(formatting_prompts_func, batched=True)


print(train_ds)


print(val_ds)


print(val_ds[:5]["text"])


trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,  # Can make training 5x faster for short sequences.
    args=TrainingArguments(
        output_dir="data/models/All",
        run_name="qa_telcom",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_ratio=0.05,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        push_to_hub=False,
        auto_find_batch_size=True,
        num_train_epochs=3,
        save_total_limit=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to="wandb",
        data_seed=41,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        save_only_model=True,
    ),
)


# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


trainer.evaluate()


trainer_stats = trainer.train()


print(trainer_stats.metrics)


trainer.evaluate()


# trainer.evaluate(test_ds, metric_key_prefix="test")
