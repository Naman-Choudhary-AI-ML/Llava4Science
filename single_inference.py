from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

# Path to your fine-tuned model
fine_tuned_model_path = "/user_data/amulyam/Projects/LLaVA/output/merged_model"

# Load the fine-tuned model
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=fine_tuned_model_path,
    model_base=None,  # Adjust if necessary based on your training configuration
    model_name=get_model_name_from_path(fine_tuned_model_path)
)

# Evaluation setup
prompt = "List all anatomical locations related to chest tube?"
image_file = "/user_data/amulyam/Projects/EHRXQA/ehrxqa/study_images/s50008188/51a7f61e-8c892b2d-a140e4f2-20f2f494-a1eea041.jpg"
# Set up evaluation arguments
args = type('Args', (), {
    "model_path": fine_tuned_model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(fine_tuned_model_path),
    "query": prompt,
    "conv_mode": None,
    "image_file": image_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()
# Perform evaluation with the fine-tuned model
eval_model(args)