# MedGuide-FT: LoRA vs QLoRA Comparative Fine-Tuning — Home Health Assistant

A comparative fine-tuning project demonstrating **LoRA** and **QLoRA** on two different open-source LLMs, applied to a synthetically generated home-health Q&A dataset. Dataset generation runs **locally** via Groq's API; both fine-tuning runs and the before/after evaluation run on **Google Colab's free tier (T4 GPU)**.

## What this project demonstrates

- Generating a custom instruction-tuning dataset from scratch using an LLM (Groq's `llama-3.1-8b-instant`), run locally
- Fine-tuning with **LoRA** (full precision adapters) on a 3B model, run on Colab
- Fine-tuning with **QLoRA** (4-bit quantized adapters) on a 7B model, run on Colab
- Fitting both fine-tuning runs within **free-tier Colab hardware** (16GB T4 GPU)
- Structured before/after evaluation comparing base vs. fine-tuned model outputs
- Practical, hands-on debugging of real dependency/version issues encountered along the way

## Project structure

```
├── generate_dataset.py          # Dataset generator — run LOCALLY (Groq API)
├── LoRA_QLoRA_Comparative_Fine_Tuning_Home_Health_Assistant.ipynb
│                                 # Run in GOOGLE COLAB — contains LoRA
│                                 # training, QLoRA training, and the
│                                 # before/after comparison, as separate
│                                 # sections within one notebook
├── dataset/
│   ├── raw_all.jsonl             # All generated Q&A pairs (flat format, for review)
│   ├── train.jsonl               # Training split (chat-message format)
│   └── val.jsonl                 # Validation split (chat-message format)
└── README.md
```

This project uses two environments:

- **`generate_dataset.py` runs locally** (not in Colab) — it calls the Groq API to synthetically generate the Q&A dataset, using a local `.env` file for the API key. The output (`train.jsonl` / `val.jsonl`) is then uploaded to Google Drive for the next stage.
- **The `.ipynb` notebook runs in Google Colab** — it handles LoRA fine-tuning, QLoRA fine-tuning, and the before/after comparison, all as separate sections in one notebook, using the dataset files from Drive. Model checkpoints and adapters are also saved back to Drive so the notebook can be resumed across separate Colab runtimes.

## Models used

| | Base Model | Technique | Precision |
|---|---|---|---|
| Notebook section: LoRA fine-tuning | `Qwen/Qwen2.5-3B-Instruct` | LoRA | fp16/bf16 |
| Notebook section: QLoRA fine-tuning | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | QLoRA | 4-bit (nf4) |

Two different model sizes were deliberately chosen to reflect a real constraint: full-precision LoRA on a 7B model doesn't reliably fit on a free-tier T4 (16GB VRAM), while 4-bit QLoRA does. Pairing a smaller model with LoRA and a larger model with QLoRA demonstrates both techniques within the same hardware budget.

## Dataset

### Generation approach

The dataset was synthetically generated using `llama-3.1-8b-instant` via the Groq API. A Python script (`generate_dataset.py`) prompted the model to produce Q&A pairs across 15 common everyday-illness topics (fever, cough, headache, sore throat, mild digestive issues, etc.), with an instruction to keep answers at a **general home-care level** — no specific drug names or dosages, and a consistent recommendation to see a doctor for severe or persistent symptoms.

Key design choices:
- **Batched generation** (small batches per API call) with a running deduplication check, to stay under Groq's free-tier rate limits and avoid repetitive questions
- **Incremental saving** to disk during generation, so a long-running batch job survives interruptions
- Each entry saved directly in **chat-message JSONL format**, ready for `SFTTrainer`/PEFT without extra conversion

### Dataset stats

The dataset was generated in **two runs**. The first run used a lower `CALLS_PER_TOPIC` setting and produced only **248 unique pairs** after deduplication — too small to be a meaningful training set for the showcase goal. The generator script supports resuming (it loads any existing `raw_all.jsonl` and skips regenerating duplicate questions), so a second run was done with a higher `CALLS_PER_TOPIC`, topping the dataset up to **550 total unique pairs** without discarding or regenerating the first run's data.

| | Count |
|---|---|
| First run (initial attempt) | 248 unique pairs |
| Second run (topped up, resumed from first) | 550 unique pairs (final) |
| Training set (`train.jsonl`) | 495 |
| Validation set (`val.jsonl`) | 55 |
| Split ratio | 90 / 10 |

### Format

Each record follows the chat-message structure used by `trl`'s `SFTTrainer`:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful home-health assistant..."},
    {"role": "user", "content": "What should I do to care for a minor cut on an adult's face?"},
    {"role": "assistant", "content": "To care for a minor cut..."}
  ]
}
```

## Training configuration

Both scripts share the same LoRA hyperparameters for a fair comparison:

| Parameter | Value |
|---|---|
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj (all linear layers) |
| Learning rate | 2e-4 |
| LR scheduler | Cosine, 5% warmup |
| Epochs | 3 |
| Effective batch size | 16 (via gradient accumulation) |

## Results

### Training metrics

| Metric | LoRA (Qwen2.5-3B) | QLoRA (DeepSeek-R1-Distill-7B) |
|---|---|---|
| Final training loss | 0.448 | 0.679 |
| Final validation loss | 0.528 | 0.788 |
| Mean token accuracy | **83.7%** | 79.7% |
| Total training steps | 93 | 93 |
| Training time (Colab free T4) | ~50 minutes | ~1 hour 45 minutes |
| Trainable parameters | 29.9M (0.96% of 3.1B) | — (comparable ratio via same LoRA config) |

**Interpretation:** The LoRA run on the smaller, full-precision model achieved lower loss and higher token accuracy, and trained roughly 2x faster. This reflects a real, expected trade-off: QLoRA's 4-bit quantization introduces some precision loss, and the 7B reasoning model had a larger behavioral gap to close (see below) — but QLoRA is what made fine-tuning a 7B model possible on free-tier hardware at all, which the LoRA approach could not do at that model size.

### Before / after: qualitative comparison

The most informative result isn't the loss numbers — it's what visibly changed in the model's actual answers.

#### LoRA (Qwen2.5-3B): style and safety-compliance shift

The base model tends toward long, numbered-list answers, and in several cases **named specific over-the-counter drugs** (e.g. "acetaminophen (Tylenol) or ibuprofen (Advil)") despite the system prompt explicitly instructing against this. The fine-tuned model consistently produces short, direct paragraph answers, uses **generic drug categories instead of brand names** ("a fever reducer or a pain reliever"), and reliably closes with a doctor-referral safety line.

> **Q: What should I do if I have a mild fever?**
>
> **Before:** *"...I know that acetaminophen or ibuprofen [response cut off at token limit]"*
>
> **After:** *"For a mild fever, try drinking plenty of fluids like water, tea, or clear broths to stay hydrated. You can also use a fever reducer or a pain reliever as directed on the label. Rest is also essential to help your body recover. If your fever persists beyond a few days or worsens over time, see a doctor."*

#### QLoRA (DeepSeek-R1-Distill-7B): reasoning-trace suppression

This base model is a distilled *reasoning* model — its default behavior is to "think out loud" in first person before answering (e.g. "Okay, so I have a mild fever, and I'm not sure what to do... Let me think through this step by step"), often getting cut off by the token limit before reaching any actual advice. Fine-tuning **suppressed this reasoning-trace behavior almost entirely**, producing direct, structured answers matching the training data's style — a more dramatic behavioral change than the LoRA case, since it overrides a strong pretrained tendency rather than just adjusting tone.

> **Q: How do I deal with a sore throat at home?**
>
> **Before:** *"Okay, so I need to figure out how to deal with a sore throat at home. I remember the assistant mentioned that I shouldn't worry unless my symptoms get worse... [continues reasoning, cut off before reaching concrete advice]"*
>
> **After:** *"To help alleviate a sore throat, try drinking warm liquids like tea or broth, and gargling with salt water. You can also use a humidifier to add moisture to the air, and use a throat lozenge or neti pot to soothe the throat. If your sore throat persists or is severe, see a doctor."*

Full before/after transcripts for all test questions, both models, are saved in `comparison_lora.json` and `comparison_qlora.json`.

## Challenges encountered (Google Colab)

Several real dependency and environment issues came up while running this on free-tier Colab. Documenting them here since they're common gotchas for anyone reproducing this project:

| Problem | Cause | Fix |
|---|---|---|
| `ImportError: Found an incompatible version of torchao` when calling `get_peft_model()` / `PeftModel.from_pretrained()` | Colab's pre-installed `torchao` version is older than what `peft` expects, even though it isn't actually needed for LoRA/QLoRA | `!pip uninstall -y torchao`, then **restart the runtime** and re-run all cells from the top |
| `ImportError: Using bitsandbytes 4-bit quantization requires bitsandbytes >=0.46.1` when loading the 4-bit QLoRA model | Fresh Colab sessions sometimes have an outdated or missing `bitsandbytes` install | `!pip install -U -q bitsandbytes`, then **restart the runtime** and re-run all cells from the top |
| `TypeError: SFTConfig.__init__() got an unexpected keyword argument 'max_seq_length'` | `trl` renamed this argument across versions (`max_seq_length` → `max_length`) | Avoided the version-specific kwarg entirely — set `tokenizer.model_max_length` directly instead of passing it to `SFTConfig` |
| `TypeError: SFTTrainer.__init__() got an unexpected keyword argument 'tokenizer'` | `trl` renamed this argument to `processing_class` in newer versions | Used `inspect.signature()` to detect which keyword the installed `trl` version expects, and pass the tokenizer under the correct name automatically |
| Groq API `429 rate_limit_exceeded` errors during dataset generation | Free-tier Groq caps `llama-3.1-8b-instant` at 6,000 tokens/minute; large batch requests exceeded this | Reduced `max_tokens` per call, reduced pairs requested per call, added a fixed delay between calls, and added automatic retry with backoff that reads Groq's suggested wait time from the error message |
| Low initial dataset yield (248 pairs instead of the expected up-to-720 ceiling) | Aggressive-enough deduplication plus some failed/empty generation calls reduced the effective yield | Added incremental saving with resume support, then re-ran the generator a second time to top up the dataset without losing or duplicating the first run's data |

**General lesson:** because Colab sessions don't persist installed packages between runtimes, and library APIs (`trl` especially) change between versions without warning, it's worth running a standard setup cell at the start of every fresh Colab session:

```python
!pip install -q -U transformers accelerate peft trl datasets bitsandbytes
!pip uninstall -y torchao
```

...followed by a runtime restart, before running any training or inference code.

## Known limitations

This is a **showcase/learning project**, not a production medical tool. Worth being explicit about its limitations:

- **Dataset size (550 examples)** is small by production fine-tuning standards (typically 2,000–5,000+ for reliable results). It's enough to demonstrate a clear, visible behavior shift, but not enough for robust generalization across the full space of home-health questions.
- **Synthetic data quality is not expert-reviewed.** The dataset was LLM-generated and only spot-checked manually, not verified by a medical professional. At least one minor factual imprecision was observed post-training (a fine-tuned answer paired "fever reducer" with "antihistamine," which isn't a typical fever-specific recommendation) — a realistic consequence of purely synthetic data generation.
- **`MAX_NEW_TOKENS=200` truncation** cuts off some base-model answers mid-sentence during the comparison, which somewhat exaggerates the apparent verbosity gap between base and fine-tuned outputs — a genuine effect (fine-tuned answers are consistently shorter) but partly a generation-limit artifact worth distinguishing from a pure training effect.
- **Not intended for real medical use.** Outputs are general, non-diagnostic home-care information by design, and the model explicitly avoids specific dosing — this project is a fine-tuning technique demonstration, not a healthcare product.

## Tech stack

- `transformers`, `peft`, `trl`, `bitsandbytes` (Hugging Face ecosystem)
- Groq API (`llama-3.1-8b-instant`) for dataset generation
- Google Colab (free tier, T4 GPU) for both dataset generation support and training
- Google Drive for model/adapter persistence across Colab sessions

## Setup: Groq API key

`generate_dataset.py` runs **locally** (not in Colab) and calls the Groq API, so you'll need a free API key before running it.

1. Create a free account and generate a key at [console.groq.com/keys](https://console.groq.com/keys).
2. In the project root — the same folder as `generate_dataset.py` — create a file named `.env`.
3. Add your key to it in this exact format — no quotes, no spaces around the `=`:

```dotenv
GROQ_API_KEY=your_actual_key_here
```

4. Install the loader library so the script can read it:

```bash
pip install python-dotenv
```

The script loads this automatically via `load_dotenv()` at the top of `generate_dataset.py` — no code changes needed.

**Important:** never commit your `.env` file to GitHub. Add it to `.gitignore` before pushing:

```
.env
```

If a key is ever accidentally exposed (e.g. pasted somewhere public), revoke it immediately from the Groq console and generate a new one — treat any exposed key as compromised.

## How to reproduce

**Locally:**
1. Set up your Groq API key as described above.
2. Run `generate_dataset.py` on your local machine to produce `train.jsonl` / `val.jsonl` under `dataset/`.
3. Upload the `dataset/` folder to your Google Drive (e.g. `MyDrive/llm-finetuning/dataset/`), so the Colab notebook can access it.

**In Google Colab:**
4. Open `LoRA_QLoRA_Comparative_Fine_Tuning_Home_Health_Assistant.ipynb` in Colab.
5. Set `Runtime → Change runtime type → T4 GPU`.
6. Mount Google Drive and run the **LoRA fine-tuning** section (Qwen2.5-3B).
7. In a fresh runtime, run the **QLoRA fine-tuning** section (DeepSeek-R1-Distill-7B) — keeping this separate from the LoRA run avoids unnecessary VRAM pressure from having two large models loaded at once.
8. Run the **comparison** section, once with `ADAPTER_TYPE = "lora"` and once with `ADAPTER_TYPE = "qlora"`, to generate before/after outputs for each.

Both trained adapters are saved to Google Drive as lightweight LoRA adapter files (a few hundred MB), not merged full models — load them on top of the respective base model at inference time using `peft.PeftModel.from_pretrained()`.
