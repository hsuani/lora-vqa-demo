"""VQAv2 loader: sample N rows, split train/eval, return HF Dataset."""
import re
from datasets import load_dataset


_PUNCT = re.compile(r"[^\w\s]")


def normalize_answer(ans: str) -> str:
    return _PUNCT.sub("", ans.lower()).strip()


def load_vqa_subset(sample_size: int = 5000, seed: int = 42):
    ds = load_dataset("HuggingFaceM4/VQAv2", split="validation", trust_remote_code=True)
    ds = ds.shuffle(seed=seed).select(range(sample_size))
    split = ds.train_test_split(test_size=0.2, seed=seed)
    return split["train"], split["test"]


def build_prompt(question: str) -> str:
    return f"Question: {question}\nAnswer:"


def collate_vqa(examples, processor, max_length: int = 256):
    """Format VQA examples for Qwen2-VL processor."""
    messages_batch = []
    for q, img in zip(examples["question"], examples["image"]):
        messages_batch.append([
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": build_prompt(q)},
                ],
            }
        ])

    texts = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages_batch
    ]

    batch = processor(
        text=texts,
        images=[ex["image"] for ex in zip(*[examples["image"]])],  # handled by processor
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    batch["labels"] = batch["input_ids"].clone()
    return batch
