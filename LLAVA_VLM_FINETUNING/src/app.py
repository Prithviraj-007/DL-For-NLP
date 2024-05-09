import torch
from transformers import LlavaForConditionalGeneration, BitsAndBytesConfig, AutoProcessor
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import requests
from PIL import Image
import gradio as gr
# from googletrans import Translator


# Load translation model and tokenizer
translate_model_name = "facebook/mbart-large-50-many-to-many-mmt"
translate_model = MBartForConditionalGeneration.from_pretrained(translate_model_name)
tokenizer = MBart50TokenizerFast.from_pretrained(translate_model_name)

# load the base model in 4 bit quantized
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
)

# finetuned model adapter path (Hugging Face Hub)
model_id = 'somnathsingh31/llava-1.5-7b-hf-ft-merged_model'

# merge the models
merged_model = LlavaForConditionalGeneration.from_pretrained(model_id,
                                                           quantization_config=quantization_config,
                                                           torch_dtype=torch.float16)

# create processor from base model
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

# function to translate
def translate(text, source_lang, target_lang):
    # Set source language
    tokenizer.src_lang = source_lang

    # Encode the text
    encoded_text = tokenizer(text, return_tensors="pt")

    # Force target language token
    forced_bos_token_id = tokenizer.lang_code_to_id[target_lang]

    # Generate the translation
    generated_tokens = translate_model.generate(**encoded_text, forced_bos_token_id=forced_bos_token_id)

    # Decode the translation
    translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    return translation



def translate_gtrans(sentence, src, dest):
    translator = Translator()
    output = translator.translate(sentence, dest=dest, src=src)
    return output.text

NO_INPUT_MESSAGE = 'कृपया कुछ इनपुट चित्र प्रदान करें'

# function for making inference
def ask_vlm(hindi_input_text='ये चित्र क्या दर्शाता है, विस्तार से बताएं।', image=None, url=None, google=False):
    
    # translate from Hindi to English based on model chosen
    if google:
        prompt_eng = translate_gtrans(hindi_input_text, 'hi', 'en')
    else:
        prompt_eng = translate(hindi_input_text, "hi_IN", "en_XX")
        
    prompt = "USER: <image>\n" + prompt_eng + " ASSISTANT:"
    
    # handle the inputs
    if (not image) and (not url):
        return NO_INPUT_MESSAGE
    elif not image:
        try:
            image = Image.open(requests.get(url, stream=True).raw)
        except:
            return 'कृपया एक वैध चित्र यूआरएल प्रदान करें!!!'

    # process the inputs
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    generate_ids = merged_model.generate(**inputs, max_new_tokens=250)
    decoded_response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    assistant_index = decoded_response.find("ASSISTANT:")

    # Extract text after "ASSISTANT:"
    if assistant_index != -1:
        text_after_assistant = decoded_response[assistant_index + len("ASSISTANT:"):]
        # Remove leading and trailing whitespace
        text_after_assistant = text_after_assistant.strip()
    else:
        text_after_assistant = None
    
    # based on model chosen translate
    if google:
        hindi_output_text = translate_gtrans(text_after_assistant, 'en', 'hi')
    else:
        hindi_output_text = translate(text_after_assistant, "en_XX", "hi_IN")

    return hindi_output_text 


# Define Gradio interface
input_question = gr.components.Textbox(lines=2, label="Question (Hindi)")
input_image = gr.components.Image(type="pil", label="Input Image")
input_image_url = gr.components.Textbox(label="Image URL", placeholder="Enter image URL")
output_text = gr.components.Textbox(label="Response (Hindi)")

# Create Gradio app
gr.Interface(fn=ask_vlm, inputs=[input_question, input_image, input_image_url], outputs=output_text, title="संस्कृति संवाद!", description="अपलोड करके या यूआरएल प्रदान करके हिंदी में एक प्रश्न और एक छवि दर्ज करें, और हिंदी में उत्तर प्राप्त करें।").launch(debug=True)
