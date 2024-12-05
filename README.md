<<<<<<< HEAD
<<<<<<< HEAD
LLaVA for Chest X rays
=======
LLaVA for Chest X rays and Clinical Dataset
>>>>>>> cfa667b24e1412fd860151130369ff75915a4ed4
=======
To use the LLaVA model with the ScienceQA dataset, follow these steps: 

(1) Navigate to the LLaVA directory using cd LLaVA. 

(2) Ensure the dataset is organized under ScienceQA/ with train, val, and test subdirectories. 

(3) Convert the dataset to the LLaVA conversation-style format by running the convert_sqa_to_llava.py script with the specified base directory and split (e.g., train, val, test). 

(4) For training, either download pretrained projector weights from the model zoo or run pretrain.sh to train your own, then fine-tune the model using finetune_sqa.sh. 

(5) For evaluation, perform multiple-GPU inference to generate responses on ScienceQA with the provided scripts, or use single-GPU inference with llava.eval.model_vqa_science to generate answers, followed by eval_science_qa.py to evaluate them. Ensure all paths are correctly set up for your environment. For more details, refer to the official ScienceQA and LLaVA documentation.

Scripts to run:

1. Prepare data:
python scripts/convert_sqa_to_llava.py \
    convert_to_llava \
    --base-dir /path/to/ScienceQA/data/scienceqa \
    --prompt-format "QCM-LEA" \
    --split {train,val,minival,test,minitest}

2. Multiple-GPU Inference
 python -m llava.eval.model_vqa_science \
  --model-path liuhaotian/llava-lcs558k-scienceqa-vicuna-13b-v1.3 \
  --question-file /path/to/ScienceQA/data/scienceqa/llava_test_QCM-LEA.json \
  --image-folder /path/to/ScienceQA/data/scienceqa/images/test \
  --answers-file vqa/results/ScienceQA/test_llava-13b.jsonl \
  --conv-mode llava_v1

3. Single-GPU Inference
(a) Generate Responses

python -m llava.eval.model_vqa_science \
    --model-path liuhaotian/llava-lcs558k-scienceqa-vicuna-13b-v1.3 \
    --question-file /path/to/ScienceQA/data/scienceqa/llava_test_QCM-LEA.json \
    --image-folder /path/to/ScienceQA/data/scienceqa/images/test \
    --answers-file vqa/results/ScienceQA/test_llava-13b.jsonl \
    --conv-mode llava_v1

(b) Evaluate Responses
python eval_science_qa.py \
    --base-dir /path/to/ScienceQA/data/scienceqa \
    --result-file vqa/results/ScienceQA/test_llava-13b.jsonl \
    --output-file vqa/results/ScienceQA/test_llava-13b_output.json \
    --output-result vqa/results/ScienceQA/test_llava-13b_result.json

>>>>>>> 7188fd7c4876a6cc6de11e89a599d37de6721020
