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


def get_answer(example) -> str:
    """VQAv2 gold answer: prefer multiple_choice_answer, fall back to first annotator."""
    ans = example.get("multiple_choice_answer")
    if ans:
        return ans
    answers = example.get("answers") or []
    if answers:
        return answers[0]["answer"]
    return ""


class QwenVQACollator:
    """Collate VQAv2 examples for Qwen2-VL supervised fine-tuning.

    Builds a 2-turn chat (user: image+question / assistant: answer), runs the
    Qwen2-VL processor over the batch, and masks every non-answer token in the
    labels so loss is only computed on the answer span.
    """

    def __init__(self, processor, max_length: int = 512):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, examples):
        images, full_texts, prompt_texts = [], [], []

        for ex in examples:
            question = ex["question"]
            answer = get_answer(ex)
            images.append(ex["image"])

            user_turn = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": ex["image"]},
                    {"type": "text", "text": build_prompt(question)},
                ],
            }]
            # Prompt only (for measuring how many tokens to mask).
            prompt_texts.append(
                self.processor.apply_chat_template(
                    user_turn, tokenize=False, add_generation_prompt=True
                )
            )
            # Prompt + assistant answer (the actual training target).
            full = user_turn + [{
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            }]
            full_texts.append(
                self.processor.apply_chat_template(
                    full, tokenize=False, add_generation_prompt=False
                )
            )

        batch = self.processor(
            text=full_texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()
        # Mask pad tokens.
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100
        # Mask image-placeholder tokens.
        image_token_id = getattr(self.processor.tokenizer, "image_token_id", None)
        if image_token_id is None:
            image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        if image_token_id is not None and image_token_id >= 0:
            labels[batch["input_ids"] == image_token_id] = -100
        # Mask the prompt span per sample (only train on the answer).
        prompt_lens = [
            len(self.processor.tokenizer(p, add_special_tokens=False)["input_ids"])
            for p in prompt_texts
        ]
        for i, plen in enumerate(prompt_lens):
            labels[i, :plen] = -100

        batch["labels"] = labels
        return batch
