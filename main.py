import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

# Set the device to CUDA if available, else CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load the image of the fruit
image = load_image("URL_to_your_fruit_image.jpg")  # Replace with the actual URL or local path to fruit image

# Initialize the processor and model
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceTB/SmolVLM-Instruct",
    torch_dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
).to(DEVICE)

# Create the input message asking if the fruit is bad
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Is this fruit ripe or has it gone bad?"}
        ]
    },
]

# Apply the chat template to generate the prompt
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

# Prepare the inputs by tokenizing the prompt and processing the image
inputs = processor(text=prompt, images=[image], return_tensors="pt")
inputs = inputs.to(DEVICE)

# Generate the output by passing the inputs to the model
generated_ids = model.generate(**inputs, max_new_tokens=500)

# Decode the generated output to text
generated_texts = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)

# Print the generated description (which will tell if the fruit is ripe or not)
print(generated_texts[0])
