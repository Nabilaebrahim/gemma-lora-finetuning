# Gemma LoRA Fine-Tuning on Google Colab

> Fine-tune Google's **Gemma 2B** language model using **LoRA** (Low-Rank Adaptation) and the **Databricks Dolly 15k** instruction dataset — all on a free Google Colab GPU.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Background: LoRA & Quantization](#background-lora--quantization)
3. [Dataset](#dataset)
4. [Prerequisites](#prerequisites)
5. [Required Credentials](#required-credentials)
   - [Kaggle API Key](#kaggle-api-key)
   - [Google / Colab Setup](#google--colab-setup)
6. [Environment Setup on Google Colab](#environment-setup-on-google-colab)
7. [Running the Fine-Tuning Script](#running-the-fine-tuning-script)
8. [Known Issues & Lessons Learned](#known-issues--lessons-learned)
9. [Solution: Lamini Platform](#solution-lamini-platform)
10. [Project Structure](#project-structure)


---

## Project Overview

This project demonstrates how to fine-tune **Gemma 2B** — Google's open-weight large language model — on an instruction-following dataset using the **LoRA** parameter-efficient fine-tuning technique.

The entire workflow is designed to run on **Google Colab** using a free or paid T4/A100 GPU, keeping resource consumption manageable through LoRA (which trains only ~1.3 million parameters instead of the full ~2.5 billion).

**What the script does, step by step:**

| Step | Description |
|------|-------------|
| 1 | Configure Kaggle and Keras environment variables |
| 2 | Install `keras-nlp` and `keras >= 3` |
| 3 | Set JAX as the Keras backend and configure XLA memory |
| 4 | Load and preprocess the Databricks Dolly 15k JSONL dataset |
| 5 | Load the pre-trained `gemma_2b_en` model via Keras NLP |
| 6 | Run a baseline inference test (before fine-tuning) |
| 7 | Enable LoRA adapters (rank = 4) — trainable params drop to ~1.3M |
| 8 | Configure the AdamW optimizer and compile the model |
| 9 | Train for 1 epoch with batch size 1 |
| 10 | Run an inference test after fine-tuning to compare results |

---

## Background: LoRA & Quantization

### What is LoRA?

**LoRA (Low-Rank Adaptation)** is a parameter-efficient fine-tuning technique introduced in the paper [*LoRA: Low-Rank Adaptation of Large Language Models*](https://arxiv.org/abs/2106.09685) (Hu et al., 2021).

Instead of updating all weights in a model during fine-tuning, LoRA **freezes the original pre-trained weights** and injects small, trainable **low-rank decomposition matrices** into each targeted layer. Concretely, for a weight matrix `W`, LoRA learns two smaller matrices `A` and `B` such that the effective weight update is `ΔW = A × B`.

**Key benefits:**

- Reduces the number of trainable parameters by **99%+** (from ~2.5B to ~1.3M for Gemma 2B at rank 4).
- Requires **far less GPU memory** — making it feasible on consumer hardware and free Colab tiers.
- The original model weights remain unchanged, so the base model can be reused for other tasks.
- Fine-tuned LoRA adapters are tiny files (a few MB) that can be merged back or served separately.

**Rank hyperparameter:** A higher rank (`r`) means more expressive adapters but higher memory use. This project uses `rank=4`, which is a common starting point for instruction tuning.

---

### What is Quantization?

**Quantization** is a model compression technique that reduces the numerical precision of model weights — for example, from 32-bit floating point (`float32`) to 8-bit integers (`int8`) or 4-bit integers (`int4`).

**Why it matters:**

- A full-precision Gemma 2B model requires roughly **~5 GB of VRAM** in `float32` or **~2.5 GB** in `float16/bfloat16`.
- 4-bit quantization (e.g., via `bitsandbytes` or `GPTQ`) can compress that down to **~1–1.5 GB**, making models runnable on 4–8 GB GPUs.
- Quantization can be applied at **inference time** (post-training quantization) or during fine-tuning (QLoRA = Quantization + LoRA).

**Note:** This project uses **bfloat16/JAX** precision through the Keras/JAX backend rather than explicit int4/int8 quantization. Full QLoRA (4-bit quantization + LoRA adapters) is a natural next step if GPU memory remains a bottleneck.

---

## Dataset

**Databricks Dolly 15k** — [`databricks-dolly-15k`](https://huggingface.co/datasets/databricks/databricks-dolly-15k)

| Property | Details |
|----------|---------|
| Format | JSONL (one JSON object per line) |
| Size | ~13 MB, ~15,000 instruction-response pairs |
| Fields used | `instruction`, `response`, `context` |
| Filtering | Records with a non-empty `context` field are skipped |
| Subset used | First **1,000 records** (for Colab demo purposes) |

Each record is formatted into the following prompt template before training:

```
Instruction:
{instruction}

Response:
{response}
```

### How to obtain the dataset

The dataset is available on **Kaggle**:

```
https://www.kaggle.com/datasets/databricks/databricks-dolly-15k
```

You need a valid **Kaggle API key** (see [Required Credentials](#required-credentials)) to download it programmatically.

---

## Prerequisites

### Hardware

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| GPU VRAM | 15 GB (T4 on Colab) | 40 GB (A100 on Colab Pro) |
| System RAM | 12 GB | 25 GB |
| Disk space | ~5 GB (model + dataset) | 10 GB |

> **Important:** The Gemma 2B model checkpoint alone is approximately **5 GB**. Ensure your Colab runtime (or local machine) has sufficient disk space before proceeding.

### Software

| Package | Version |
|---------|---------|
| Python | 3.10+ |
| keras | >= 3.0 |
| keras-nlp | latest stable |
| JAX | installed automatically with `keras-nlp` |
| tensorflow / torch | one backend required by Keras 3 |

> All packages are installed inside the Colab notebook — no local installation is required.

### Accounts

- A **Google account** with access to [Google Colab](https://colab.research.google.com)
- A **Kaggle account** with an API key
- Access to [Google's Gemma model on Kaggle](https://www.kaggle.com/models/google/gemma) — you must accept the model's terms of use

---

## Required Credentials

### Kaggle API Key

The Kaggle API key is required to:
1. Download the **Databricks Dolly 15k** dataset.
2. Download the **Gemma 2B** model weights via `keras_nlp.models.GemmaCausalLM.from_preset()`.

**Steps to obtain your Kaggle API key:**

1. Log in to [https://www.kaggle.com](https://www.kaggle.com).
2. Click your profile picture → **Settings**.
3. Scroll to the **API** section and click **Create New Token**.
4. A file named `kaggle.json` will be downloaded. It contains:
   ```json
   {
     "username": "your_kaggle_username",
     "key": "your_kaggle_api_key"
   }
   ```
5. Accept the Gemma model licence at [https://www.kaggle.com/models/google/gemma](https://www.kaggle.com/models/google/gemma).

**Setting credentials in the script:**

Replace the placeholder values in [`gemma_lora_finetuning.py`](gemma_lora_finetuning.py) (Cell 1) with your actual credentials — or, preferably, set them as Colab Secrets (see below):

```python
os.environ["KAGGLE_USERNAME"] = "your_kaggle_username"
os.environ["KAGGLE_KEY"]      = "your_kaggle_api_key"
```

> **Security:** Never commit your real credentials to version control. The `.gitignore` already excludes `kaggle.json` and `.env` files. Use **Colab Secrets** or environment variables at runtime only.

---

### Google / Colab Setup

1. Open [https://colab.research.google.com](https://colab.research.google.com) and sign in with your Google account.
2. Create a new notebook or upload `gemma_lora_finetuning.py`.
3. Set the runtime to **GPU**:
   - **Runtime → Change runtime type → Hardware accelerator → GPU (T4 or A100)**
4. *(Recommended)* Store credentials as **Colab Secrets** instead of hardcoding them:
   - Click the 🔑 **Secrets** icon in the left sidebar.
   - Add `KAGGLE_USERNAME` and `KAGGLE_KEY` as secrets.
   - In your code, retrieve them securely:
     ```python
     from google.colab import userdata
     os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
     os.environ["KAGGLE_KEY"]      = userdata.get("KAGGLE_KEY")
     ```

---

## Environment Setup on Google Colab

Run the following cells in order inside your Colab notebook:

### Step 1 — Set credentials

```python
import os
from google.colab import userdata

os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"]      = userdata.get("KAGGLE_KEY")
```

### Step 2 — Install dependencies

```python
!pip install -q -U keras-nlp
!pip install -q -U "keras>=3"
```

### Step 3 — Configure the Keras backend

```python
os.environ["KERAS_BACKEND"]                = "jax"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"
```

> Setting `XLA_PYTHON_CLIENT_MEM_FRACTION=1.0` allows JAX to pre-allocate all available GPU memory, reducing fragmentation during training.

### Step 4 — Download the dataset from Kaggle

```bash
!kaggle datasets download -d databricks/databricks-dolly-15k --unzip
```

### Step 5 — Run the script

```bash
!python gemma_lora_finetuning.py
```

Or paste the cell contents directly into your Colab notebook and run them sequentially.

---

## Running the Fine-Tuning Script

The script [`gemma_lora_finetuning.py`](gemma_lora_finetuning.py) is structured as 10 clearly labelled cells that map 1-to-1 with a Colab notebook workflow:

```
Cell 1  → Set environment variables (Kaggle credentials, backend)
Cell 2  → Install keras-nlp and keras >= 3
Cell 3  → Set JAX backend + XLA memory flag
Cell 4  → Load & filter Databricks Dolly 15k (first 1000 records)
Cell 5  → Load gemma_2b_en from Kaggle via keras_nlp
Cell 6  → Baseline inference test (before fine-tuning)
Cell 7  → Enable LoRA (rank=4) — ~1.3M trainable params
Cell 8  → Compile with AdamW (lr=5e-5, weight_decay=0.01)
Cell 9  → Train: 1 epoch, batch_size=1
Cell 10 → Post-fine-tuning inference test
```

**Expected training time** on a Colab T4 GPU: approximately **60–120 minutes** for 1000 samples at batch size 1.

---

## Known Issues & Lessons Learned

### `JaxRuntimeError: RESOURCE_EXHAUSTED` — Out of GPU Memory

During development, the following error was encountered when calling `gemma_lm.fit()` on a **Colab T4 GPU (15 GB VRAM)**:

```
WARNING:absl:prefetch_buffer_size=8 is smaller than num_threads=16.
This will limit the number of threads that can actually be used in parallel to 8,
potentially hurting performance.

JaxRuntimeError                           Traceback (most recent call last)
/tmp/ipykernel_1130/2244755920.py in <cell line: 0>()
----> 1 gemma_lm.fit(
      2     data,
      3     epochs=1,
      4     batch_size=1
      5 )

... (7 frames) ...

/usr/local/lib/python3.13/dist-packages/jax/_src/array.py in _value(self)
    640     # addressable_device_list can be empty. If it's empty, we will error below
    641     if self.is_fully_replicated and self.sharding.has_addressable_devices:
--> 642         npy_value, did_copy = self._single_device_array_to_np_array_did_copy()
    643         npy_value.flags.writeable = False
    644         if did_copy:

JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 1.95GiB
with allocator GPU_0_bfc on device 0.
[executable_name='jit_greater'] [tf-allocator-allocation-error='']
```

**Root cause:** Even with `XLA_PYTHON_CLIENT_MEM_FRACTION=1.0` and `batch_size=1`, loading the full Gemma 2B model in JAX on a 15 GB T4 leaves insufficient memory for the forward pass, loss computation, and gradient accumulation at a sequence length of 512 tokens.

**Things attempted before finding a solution:**

- Reducing `sequence_length` to 128 and 256
- Reducing batch size to 1 (already minimum)
- Restarting the runtime to clear memory fragmentation
- Switching to the TensorFlow and PyTorch backends

None of these fully resolved the OOM on the free Colab T4.

---

## Solution: Lamini Platform

After encountering persistent `RESOURCE_EXHAUSTED` OOM errors on the Colab T4, the fine-tuning workflow was migrated to the **[Lamini Platform](https://lamini.ai)**.

Lamini provides a managed fine-tuning API that:

- Handles all GPU memory management server-side (no local VRAM constraints).
- Supports instruction fine-tuning with a simple Python API:
  ```python
  from lamini import Lamini
  llm = Lamini(model_name="meta-llama/Meta-Llama-3-8B-Instruct")
  llm.train(data)
  ```
- Eliminates the need to manage Keras/JAX backend configuration or XLA memory flags.
- Is free to try with an API key from [https://app.lamini.ai](https://app.lamini.ai).

> **Takeaway:** For Gemma 2B (and larger) fine-tuning on free-tier Colab GPUs, memory constraints are a hard blocker with the JAX/Keras stack. Either use **Colab Pro with an A100**, apply **4-bit QLoRA** via `bitsandbytes`, or use a managed platform like **Lamini**.

---

## Project Structure

```
gemma_lora_finetuning/
├── gemma_lora_finetuning.py   # Main fine-tuning script (10 cells)
├── .gitignore                 # Excludes secrets, model weights, datasets
└── README.md                  # This file
```

**Files intentionally excluded from the repository** (see [`.gitignore`](.gitignore)):

| Excluded | Reason |
|----------|--------|
| `kaggle.json` | Contains secret API credentials |
| `*.jsonl` | Dataset files — download via Kaggle CLI |
| `*.h5`, `*.safetensors`, `*.bin` | Model weight files — too large for Git |
| `.env` | Environment variable files containing secrets |


