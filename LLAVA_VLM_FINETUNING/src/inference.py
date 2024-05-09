import torch
from transformers import LlavaForConditionalGeneration, BitsAndBytesConfig, AutoProcessor
from peft import PeftModel
import requests
from PIL import Image


def load_base_model(model_id):
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    base_model = LlavaForConditionalGeneration.from_pretrained(model_id,
                                                               quantization_config=quantization_config,
                                                               torch_dtype=torch.float16)
    return base_model


def load_peft_lora_adapter(base_model, peft_lora_adapter_path):
    peft_lora_adapter = PeftModel.from_pretrained(base_model, peft_lora_adapter_path, adapter_name="lora_adapter")
    return peft_lora_adapter


def merge_adapters(base_model, peft_lora_adapter_path):
    base_model.load_adapter(peft_lora_adapter_path, adapter_name="lora_adapter")
    return base_model


def main():
    model_id = "llava-hf/llava-1.5-7b-hf"  # Actual base model id
    peft_lora_adapter_path = 'somnathsingh31/llava-1.5-7b-hf-ft-museum'  # Actual adapter path

    # Load the base model
    base_model = load_base_model(model_id)

    # Load the PEFT Lora model (adapter)
    peft_lora_adapter = load_peft_lora_adapter(base_model, peft_lora_adapter_path)

    # Merge the adapters into the base model
    merged_model = merge_adapters(base_model, peft_lora_adapter_path)

    prompt = "USER: <image>\nWhat is special in this chess set and pieces? \nASSISTANT:"
    url = "https://images.metmuseum.org/CRDImages/ad/original/138425.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    processor = AutoProcessor.from_pretrained(model_id)
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    
    # ... process the image and create inputs ...
    generate_ids = merged_model.generate(**inputs, max_new_tokens=150)
    decoded_response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print("Generated response:", decoded_response)


if __name__ == "__main__":
    main()
