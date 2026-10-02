# Rethinking the Effectiveness of Contrastive Decoding in Mitigating Hallucinations in MLLMs

[![NeurIPS 2026 Workshop](https://img.shields.io/badge/NeurIPS%202026-VLM4RWD%20Workshop-8A2BE2)](#citation)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Models](https://img.shields.io/badge/models-LLaVA--1.5%20%7C%20Qwen2.5--VL%20%7C%20InternVL3-orange)

Official code for **"Rethinking the Effectiveness of Contrastive Decoding in Mitigating Hallucinations in MLLMs"**, accepted at the **NeurIPS 2026 VLM4RWD Workshop**.

Contrastive decoding (CD) is a popular training-free fix for object hallucination in multimodal LLMs. The idea is to contrast a normal forward pass (the *expert*) against a hallucination-prone one (the *amateur*) and subtract out the hallucinated content. We check whether the reported benchmark gains actually come from that mechanism. **They don't.**

## TL;DR

A correction that suppresses hallucination has to *depend on whether the model is hallucinating*, and that dependence can be measured inside the model. We measure it for three CD methods (**VCD**, **ICD**, **SID**) on three models (**LLaVA-v1.5-7B**, **Qwen2.5-VL-7B-Instruct**, **InternVL3-8B**), across discriminative (POPE) and generative (CHAIR, LLaVA-Bench) tasks, and find:

1. **No selectivity in discriminative settings.** The contrastive shift is unrelated to hallucination status at every layer, even where object presence is linearly decodable (from layer 8 onward). The amateur still encodes object presence, so that signal cancels when you subtract.
2. **Gains come from greedy decoding.** In free-form captioning, the Adaptive Plausibility Constraint (APC) pushes sampling towards greedy decoding. Plain greedy decoding beats every CD method on every model.
3. **No hallucination-specific signal.** The contrastive adjustment *amplifies* hallucinated object tokens more often than it suppresses them. Replacing it with matched Gaussian noise does as well or better.

> The gains reported on standard hallucination benchmarks reflect a change in decoding behaviour, not hallucination mitigation. Inference-time methods should be validated against **greedy decoding** and **structureless controls** before their gains are accepted.

## Background

At step *t*, contrastive decoding scores each token as

```
z_cd = (1 + α) · z(y | v, x) − α · z(y | v*, x*)
```

where `(v*, x*)` is a perturbed input meant to amplify the language prior. With α = 1 this is `z_cd = E + d_t`, where `d_t = E − A` is the **contrastive adjustment** (expert minus amateur logits). The three methods differ only in how the amateur is built:

| Method | Amateur construction |
|---|---|
| **VCD** (Visual CD) | Image corrupted with forward-diffusion noise |
| **ICD** (Instruction CD) | Misleading role instruction prepended to the prompt |
| **SID** (Self-Introspective Decoding) | Keeps only the least informative visual tokens (picked by early-layer attention) |

All three are paired with the **APC**, which masks any token whose expert probability is below `β · max p`.

## Key results

### 1. The contrastive shift ignores hallucination status (POPE, logit lens)

LLaVA-v1.5-7B on POPE-COCO (9,000 questions). Selectivity AUC = probability that a hallucinated "Yes" is pushed away from "Yes" more than a correct "Yes" (0.5 = chance).

| Method | Selectivity AUC [95% CI] | Mean shift Δ on H | Mean shift Δ on T | Amateur presence AUC | % of H pushed *towards* "Yes" |
|---|---|---|---|---|---|
| VCD | 0.523 [0.49, 0.55] | +3.32 | +3.60 | 0.777 | 96.5% |
| ICD | 0.639 [0.60, 0.67] | −0.17 | −0.06 | 0.940 | 25.7% |
| SID | 0.691 [0.66, 0.72] | +1.61 | +3.11 | 0.799 | 79.7% |

Controls show this is a real absence, not a blunt test. An oracle selective shift reaches AUC ≈ 0.89 under the same computation. Each method's shift is 9–24× larger than the smallest selective effect the test can detect. And **a single constant bias added to every question (b\* = +0.84) gets 86.86% POPE accuracy**, beating every CD method (best: 85.89%) without using any image information.

### 2. Greedy decoding accounts for the gains (CHAIR on 500 MS COCO images, lower is better)

| Condition | LLaVA-v1.5 CHAIR-S | LLaVA-v1.5 CHAIR-I | Qwen2.5-VL CHAIR-S | Qwen2.5-VL CHAIR-I | InternVL3 CHAIR-S | InternVL3 CHAIR-I |
|---|---|---|---|---|---|---|
| Sample | 57.4 | 17.3 | 31.2 | 10.4 | 33.5 | 9.1 |
| VCD (β=0.1) | 57.2 | 17.4 | 28.8 | 8.9 | 33.5 | 9.5 |
| SID (β=0.1) | 51.8 | 15.0 | 29.6 | 9.9 | 33.2 | 9.3 |
| APC-S (β=0.1, no contrast) | 53.4 | 16.3 | 23.5 | 7.5 | 29.7 | 8.4 |
| **Greedy** | **50.0** | **13.9** | **22.8** | **7.3** | **29.4** | **8.1** |

On LLaVA-Bench (In-the-Wild), the Jaccard similarity between APC-constrained sampling and greedy output rises steadily with β (LLaVA: 0.34 → 0.97), while similarity to plain sampling stays flat.

### 3. A Gaussian proxy matches or beats the real adjustment (LLaVA-v1.5-7B, CHAIR)

| Method | CHAIR-S ↓ | CHAIR-I ↓ |
|---|---|---|
| VCD (β=0.1) | 57.2 | 17.4 |
| Proxy N(0.026, 0.380²) | **51.4** | **14.1** |
| SID (β=0.1) | 51.8 | 15.0 |
| Proxy N(0.088, 0.851²) | **51.5** | **15.0** |

## Repository structure

Each experiment folder is self-contained and has its own README with setup and run instructions.

```
Rethinking-CD/
├── pope repro files/            # POPE reproduction of baseline/VCD/ICD/SID (+ PBA, OLM, APC)
│   ├── LLava1.5-7B/
│   ├── qwen2.5-7B/
│   └── inference shell scripts/
├── mme files/                   # MME benchmark evaluation of the same decoding methods
├── label_imbalance_experiment/  # POPE re-scored under 25/75 and 75/25 label ratios
├── mech_interp/                 # Logit-lens + linear-probe analysis on POPE-COCO (Sec. 5.1)
├── jaccard_experiment/          # APC-vs-greedy/sampling Jaccard sweep on LLaVA-Bench (Sec. 5.2)
│   ├── jaccard_on_llava/
│   └── jaccard_on_qwen/
├── CHAIR_analysis/              # CHAIR scores, contrastive adjustment d = E − A, Gaussian proxy (Sec. 5.3–5.4)
│   ├── agreement_and_overlap/
│   └── contrastive_adjustment/
└── intern/                      # InternVL3-8B ports of the experiments above
```

| Paper section | Folder |
|---|---|
| §5.1 Contrastive shift is not selective (logit lens, probes, oracle, scalar-bias sweep) | `mech_interp/` |
| §5.2 APC shifts generation towards greedy decoding | `jaccard_experiment/`, `intern/jaccard_experiment/` |
| §5.3 Greedy decoding accounts for the reported gains | `CHAIR_analysis/` |
| §5.4 Contrastive adjustment carries no hallucination signal + Gaussian proxy | `CHAIR_analysis/contrastive_adjustment/` |
| POPE / MME reproduction and label-imbalance stress test | `pope repro files/`, `mme files/`, `label_imbalance_experiment/` |

## Getting started

Every experiment runs on **a single 24 GB GPU** (all paper results were produced on one NVIDIA L4).

```bash
git clone https://github.com/Aurnawr/Rethinking-CD.git
cd Rethinking-CD
```

Dependencies differ between experiments. The LLaVA code paths pin older `transformers`/`torch` versions, and the Qwen and InternVL paths need recent `transformers` (InternVL3 needs `>=4.52`). Set up an environment per experiment by following its README, for example:

```bash
# CHAIR analysis
cd CHAIR_analysis
pip install -r requirements.txt
bash download_assets.sh
bash contrastive_adjustment/llava-1.5-7b/run.sh

# Mechanistic (logit-lens) analysis: smoke test first, then the full run
cd mech_interp
MAX_SAMPLES=40 bash run_all.sh
bash run_all.sh

# Jaccard sweep
cd jaccard_experiment/jaccard_on_llava
export MODEL_PATH=/path/to/llava-v1.5-7b
bash scripts/run_all.sh
```

### Default settings

| Setting | Value |
|---|---|
| Contrastive strength α | 1 |
| APC threshold β | 0.1 (swept over [0, 1] for the Jaccard analysis) |
| Sampling temperature τ | 1.0 |
| Max new tokens | 1024 |
| CHAIR prompt | "Describe this image in detail." |
| CHAIR images | 500 MS COCO val images (`CHAIR_analysis/image_ids_500.json`) |
| Confidence intervals | Image-level cluster bootstrap, 95%, B = 10,000 |

## Benchmarks and models

- **Benchmarks:** POPE (COCO / A-OKVQA / GQA × random / popular / adversarial), CHAIR (MS COCO), LLaVA-Bench (In-the-Wild), MME
- **Models:** [LLaVA-v1.5-7B](https://huggingface.co/liuhaotian/llava-v1.5-7b), [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), [InternVL3-8B](https://huggingface.co/OpenGVLab/InternVL3-8B-hf)
- **CD methods:** [VCD](https://arxiv.org/abs/2311.16922), [ICD](https://arxiv.org/abs/2403.18715), [SID](https://arxiv.org/abs/2408.02032), each run from its official implementation

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{rethinkingcd2026,
  title     = {Rethinking the Effectiveness of Contrastive Decoding in Mitigating Hallucinations in MLLMs},
  author    = {TODO: author list},
  booktitle = {NeurIPS 2026 Workshop on VLM4RWD},
  year      = {2026}
}
```

## Acknowledgements

This work builds on the reproducibility critique of [Yin et al. (2026), *The Mirage of Performance Gains*](https://arxiv.org/abs/2504.10020), and on the open-source code of [LLaVA](https://github.com/haotian-liu/LLaVA), [VCD](https://github.com/DAMO-NLP-SG/VCD), [ICD](https://arxiv.org/abs/2403.18715), [SID](https://github.com/huofushuo/SID), [POPE](https://github.com/RUCAIBox/POPE), and [CHAIR](https://arxiv.org/abs/1809.02156).
