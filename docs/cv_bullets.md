# CV bullets — LoRA VQA demo

Paste-ready. Pick the variant that fits the role; keep one, not all.

## Parallel / Side-Project section

**Concise (1 line)**
- Fine-tuned **Qwen2-VL-2B** on VQAv2 with **LoRA (PEFT)**; ran an 8-point
  rank×alpha sweep (r∈{4,8,16,32}, α∈{r,2r}) on a single L4, quantifying the
  capacity–accuracy trade-off (≤0.2% trainable params, 0.699→0.713 exact match).

**Detailed (2 lines, SA / ML-systems flavor)**
- Built a reproducible LoRA fine-tuning pipeline for the **Qwen2-VL-2B** vision-language
  model on VQAv2 (HF `transformers`/`peft`/`Trainer`), with answer-only loss masking
  and a custom multimodal collator that bounds vision-token count via `max_pixels`.
- Ran a **rank/alpha sensitivity study** (8 configs, single NVIDIA L4, ~21 min/run),
  showing rank gains are sublinear — an **8× parameter increase bought only +1.4 pts**
  exact-match — and that `alpha/r = 2×` reliably beats `1×`; selected `r8_a16` as the
  best accuracy-per-parameter operating point.

## Talking points (verbal, for the interview itself)
- Why LoRA over full FT: trains <0.2% of params, no optimizer-state blow-up, swappable
  adapters — the standard way to specialize a frozen base on commodity GPUs.
- Why `q_proj`/`v_proj` only: the classic LoRA-paper target set; cheapest high-leverage
  projections in attention. Adding `k_proj`/`o_proj`/MLP is the obvious next ablation.
- The one number to remember: **+1.4 pts for 8× params** → on this data, capacity isn't
  the bottleneck; data scale / target-module coverage would move the needle more.
