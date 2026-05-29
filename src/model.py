"""Load Qwen2-VL-2B and wrap with LoRA config."""
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType


def load_model_and_processor(model_name: str = "Qwen/Qwen2-VL-2B-Instruct"):
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    return model, processor


def apply_lora(model, r: int, lora_alpha: int, lora_dropout: float = 0.05,
               target_modules: list[str] | None = None):
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def count_trainable_params(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "ratio": trainable / total}
