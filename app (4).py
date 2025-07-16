import gradio as gr
from byaldi import RAGMultiModalModel
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from PIL import Image
import os
import traceback
import spaces
import re

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the Byaldi and Qwen2-VL models
rag_model = RAGMultiModalModel.from_pretrained("vidore/colpali")  # Byaldi model
qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct", trust_remote_code=True, torch_dtype=torch.bfloat16
).to(device)  # Move Qwen2-VL to GPU

# Processor for Qwen2-VL
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", trust_remote_code=True)

# Function for OCR and text extraction
# @spaces.GPU(duration=120)  # Increased GPU duration to 120 seconds
def ocr_and_extract(image):
    try:
        # Save the uploaded image temporarily
        temp_image_path = "temp_image.jpg"
        image.save(temp_image_path)

        # Index the image with Byaldi, and force overwrite of the existing index
        rag_model.index(
            input_path=temp_image_path,
            index_name="image_index",  # Reuse the same index
            store_collection_with_index=False,
            overwrite=True  # Overwrite the index for every new image
        )

        # Perform the search query on the indexed image
        results = rag_model.search("", k=1)

        # Prepare the input for Qwen2-VL
        image_data = Image.open(temp_image_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_data},
                ],
            }
        ]

        # Process the message and prepare for Qwen2-VL
        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)

        # Move the image inputs and processor outputs to CUDA
        inputs = processor(
            text=[text_input],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)

        # Generate the output with Qwen2-VL
        generated_ids = qwen_model.generate(**inputs, max_new_tokens=50)
        output_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # Filter out "You are a helpful assistant" and "assistant" labels
        filtered_output = [line for line in output_text[0].split("\n") if not any(kw in line.lower() for kw in ["you are a helpful assistant", "assistant", "user", "system"])]
        extracted_text = "\n".join(filtered_output).strip()

        # Clean up the temporary file
        os.remove(temp_image_path)

        return extracted_text

    except Exception as e:
        error_message = str(e)
        traceback.print_exc()
        return f"Error: {error_message}"

ocr_and_extract = spaces.GPU(duration=120)(ocr_and_extract)

def search_keywords(extracted_text, keywords):
    if not extracted_text:
        return "No text extracted yet. Please upload an image."

    # Highlight matching keywords in the extracted text
    highlighted_text = extracted_text
    for keyword in keywords.split():
        highlighted_text = re.sub(f"({re.escape(keyword)})", r"<mark>\1</mark>", highlighted_text, flags=re.IGNORECASE)

    # Return the highlighted text as HTML
    return f"<div style='white-space: pre-wrap'>{highlighted_text}</div>"

# Gradio interface for image input and keyword search
with gr.Blocks() as iface:
    # Add a title at the top of the interface
    gr.HTML("<h1 style='text-align: center'>Byaldi + Qwen2VL</h1>")

    # Image upload and text extraction section
    with gr.Column():
        img_input = gr.Image(type="pil", label="Upload an Image")
        extracted_output = gr.Textbox(label="Extracted Text", interactive=False)  # Use Textbox to store text

        # Functionality to trigger the OCR and extraction
        img_button = gr.Button("Extract Text")
        img_button.click(fn=ocr_and_extract, inputs=img_input, outputs=extracted_output)

    # Keyword search section
    with gr.Column():
        search_input = gr.Textbox(label="Enter keywords to search")
        search_output = gr.HTML(label="Search Results")

        # Functionality to search within the extracted text
        search_button = gr.Button("Search")
        search_button.click(fn=search_keywords, inputs=[extracted_output, search_input], outputs=search_output)

iface.launch()