"""VQA exact-match accuracy (canonical normalization)."""
import re


_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = re.compile(r"[^\w\s]")
_WHITESPACE = re.compile(r"\s+")


def vqa_normalize(s: str) -> str:
    s = s.lower()
    s = _PUNCT.sub(" ", s)
    s = _ARTICLES.sub("", s)
    s = _WHITESPACE.sub(" ", s).strip()
    return s


def exact_match(predictions: list[str], references: list[str]) -> float:
    assert len(predictions) == len(references)
    hits = sum(
        vqa_normalize(p) == vqa_normalize(r)
        for p, r in zip(predictions, references)
    )
    return hits / len(predictions)


def evaluate_model(model, processor, eval_dataset, device="cuda", max_new_tokens=20):
    import torch
    model.eval()
    preds, refs = [], []

    for example in eval_dataset:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": example["image"]},
                    {"type": "text", "text": f"Question: {example['question']}\nAnswer:"},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[example["image"]], return_tensors="pt").to(device)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens)

        decoded = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        preds.append(decoded.strip())
        refs.append(example["answers"][0]["answer"])

    return exact_match(preds, refs)
