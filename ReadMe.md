# Generating Behavior-Driven Development (BDD) Artifacts via LLMs
This repository contains the code, datasets, and evaluation results for my Master’s thesis. The research investigates the bidirectional transformation between semi-structured technical `Records` and structured Gherkin `Scenarios` using Large Language Models (LLMs).

We conduct a systematic comparison between two primary model adaptation approaches:
1. **Supervised Fine-Tuning (SFT):** Specialized training of CodeT5+ models (220M and 770M variants).
2. **Retrieval-Augmented Generation (RAG):** In-context learning using Llama-3.1-8B, DeepSeek-Coder, and CodeT5+ models in 1-shot and 3-shot configurations.

## Repository Structure

The repository is organized into three main directories:

### 1. `/Code` 

Contains the Jupyter notebooks used to execute the experiments:
- **FineTuning/:** `220m.ipynb` and `770m.ipynb` for model training and inference.
- **RAG/:** `records-rag.ipynb` and `scenarios-rag.ipynb` for retrieval-augmented generation pipelines, and `llm-judge.ipynb` for automated evaluation.

### 2. `/Dataset` 
The datasets used for the finetuning and rag experiments:
- `FineTuning/FeatureInputRecordOutput/prepare_Dataset.json`: The raw aligned pairs used for finetuning record generation
- `FineTuning/RecordInputFeatureOutput/prepare_Dataset.json`: The raw aligned pairs used for finetuning scenario generation
- `Rag/prepare_Dataset.json`:The raw aligned pairs used for Rag

### 3. `/FineTuning` 
Detailed results and data for all fine-tuning experiments, categorized by model scale (**220M** vs **770M**) and task direction.

- **GeneratingRecords:** Scenario → Record transformation.
- **GeneratingScenarios:** Record → Scenario transformation.
- **Sub-folders:** Further divided into **Prefixed** (using task identifiers) and **Unprefixed** settings.
- **Strategies:** Each folder contains the specific context-management data for:
  - *Truncating:* length cutoff where only the initial 512 tokens are preserved.
  - *Summarizing:* Using abstractive summarization to condense the entire artifact into the 512-token limit.
  - *Chunking:* One-to-many segment training.
  - *Truncating + Chunking:* A one-to-many approach where the target output is split into continuous chunks, each paired with a truncated version of the input.
  - *Summarizing + Chunking:* A one-to-many approach where target chunks are paired with a logically complete summary of the input.
- **Artifacts:** Each strategy folder contains
-  `Modified_Dataset.json:`The data after applying the specific strategy
-  `practice_dataset_codet5p_tokenized.json:`The final processed data used for training.
-  `validation_report.txt:`The automated metrics (EM, BLEU, F1) for that specific run.

### 4. `/Prompt_FewRag` 
Results for the RAG-based prompting experiments:
- **1-examples / 3-examples:** Comparison between 1-shot and 3-shot demonstrations.
- **Model Folders:** Results for CodeT5+, DeepSeek, and Llama.
- **Evaluation Artifacts:** Includes `final_evaluation_report.txt` and a dedicated LLM judge folder containing JSON-formatted reasoning and scores.

## Experimental variables
This study evaluates the impact of three primary variables on generation quality:
- **Task Direction:** Bidirectional mapping (Record ↔ Scenario).
- **Context Management:** Managing the 512-token limit via Truncation, Summarization, and Chunking.
- **Instructional Prefixing:** Testing the impact of explicit task signaling (e.g., `generate test scenario:`).

## Evaluation Methodology
We utilize a multi-layered evaluation framework to ensure result reliability:

- **Automated Metrics:** Exact Match (EM), BLEU, and token-based F1-score.
- **Human Evaluation:** A manual review by two experts using a 4-point rubric (Poor, Partial, Good, Excellent).
- **LLM-as-a-Judge:** Automated assessment using `GPT-QSS-120B` and `Qwen-2-32B` to evaluate structural accuracy and logical consistency.

**LLM Judge:** Each LLM judge folder contains
- *Sample.txt:* The primary reference file for qualitative analysis. it contains 10 samples including the Input, the Expected Output (Ground Truth), and the Generated Output (Prediction). These 10 samples were used consistently for both the human expert review and the automated LLM judge assessment to ensure a direct comparison.
- *evaluation_data.json:* The formatted data used as input for the judge models.
- *gpt_batch_results.json:* The raw output from the judge models (GPT-OSS-120B) accessed via the Cerebras API, including the numerical scores (1-4) and the qualitative reasoning for each judgment.
- *qwen_batch_results.json:* The raw output from the judge model (Qwen-3-32B) accessed via the Cerebras API, including the numerical scores (1-4) and the qualitative reasoning for each judgment.
## Requirements & Setup
The experiments were performed using Python and the Hugging Face `transformers` library.

**Hardware Environments:**

- **CodeT5p-220M:** Executed on Kaggle using NVIDIA P100 GPUs.
- **CodeT5p-770M & LLMs:** Executed on the Trillium compute cluster utilizing four NVIDIA H100 SXM GPUs.

**Key Libraries:**

- `transformers`
- `datasets`
- `sentence-transformers` (for BGE-large-en-v1.5 embeddings)
- `chromadb` (for vector storage)


