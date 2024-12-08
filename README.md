# LLAVA for Science

## Objective
The primary goal of this project is to extend LLaVA’s capabilities for scientific reasoning by fine-tuning it on curated datasets spanning diverse scientific domains. By doing so, we aim to enhance the model's ability to tackle complex scientific questions and tasks.

## Datasets
The following datasets were used for fine-tuning the model:
- **AI2D**: Dataset focusing on diagram understanding.
- **GeoQA+**: A geography-based question-answering dataset.
- **CLEVR-MATH**: Dataset combining visual reasoning and mathematical problem-solving.
- **SciGraphQA**: A dataset for scientific graph-based reasoning.

These datasets are available in the `datasets/` folder.

## Steps to Fine-Tune the Model

### 1. Prepare the Environment
Ensure all dependencies are installed and the required datasets are accessible in the `datasets/` folder.

### 2. Run the Fine-Tuning Script
Navigate to the `scripts/` folder and use the `lora_science.sh` script to fine-tune the model.

#### Things to Edit in the Script:
- `data_path`: Path to the dataset.
- `image_folder`: Path to the folder containing related images (if applicable).
- `output_dir`: Directory to save the fine-tuned model.



### 3. Run the generate_eval.sh script (in main folder) to generate answers and evaluations
- Convert validation json files to jsonl form
- Generate model answers using model_vqa.py
- Evaluate answers through GPT-4 using OPEN API key in multimodal_chatgpt_eval.py
- Evaluate answers with more scores using eval_more_metrics.py
