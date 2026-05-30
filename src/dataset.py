"""VQAv2 loader: sample N rows, split train/eval, return HF Dataset."""
import re
from datasets import load_dataset


_PUNCT = re.compile(r"[^\w\s]")


def normalize_answer(ans: str) -> str:
    return _PUNCT.sub("", ans.lower()).strip()


def load_vqa_subset(sample_size: int = 5000, seed: int = 42,
                    dataset_name: str = "lmms-lab/VQAv2"):
    # datasets>=3 dropped loading-script support, so use a parquet-native repo
    # (lmms-lab/VQAv2) instead of the script-based HuggingFaceM4/VQAv2.
    ds = load_dataset(dataset_name, split="validation")
    ds = ds.shuffle(seed=seed).select(range(min(sample_size, len(ds))))
    split = ds.train_test_split(test_size=0.2, seed=seed)
    return split["train"], split["test"]


def build_prompt(question: str) -> str:
    return f"Question: {question}\nAnswer:"


def get_answer(example) -> str:
    """VQAv2 gold answer: prefer multiple_choice_answer, fall back to first annotator.

    Handles both annotation formats: answers as list[dict{"answer": ...}]
    (original VQAv2) and answers as list[str] (some parquet mirrors).
    """
    ans = example.get("multiple_choice_answer")
    if ans:
        return ans
    answers = example.get("answers") or []
    if answers:
        first = answers[0]
        return first["answer"] if isinstance(first, dict) else first
    return ""


class QwenVQACollator:
    """Collate VQAv2 examples for Qwen2-VL supervised fine-tuning.

    Builds a 2-turn chat (user: image+question / assistant: answer), runs the
    Qwen2-VL processor over the batch, and masks every non-answer token in the
    labels so loss is only computed on the answer span.
    """

    def __init__(self, processor, max_pixels: int = 512 * 28 * 28):
        self.processor = processor
        # Cap image resolution so the number of vision tokens stays bounded.
        # Qwen2-VL emits one token per 28x28 patch; max_pixels caps that count
        # without truncating (truncating would chop image placeholders and
        # desync the text/input_ids image-token counts).
        self.max_pixels = max_pixels

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

        # No truncation: it would cut image placeholder tokens and break the
        # processor's text/input_ids image-token count check. Bound vision
        # tokens via max_pixels (set on the image processor) instead.
        if hasattr(self.processor, "image_processor"):
            self.processor.image_processor.max_pixels = self.max_pixels
        batch = self.processor(
            text=full_texts,
            images=images,
            padding=True,
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
