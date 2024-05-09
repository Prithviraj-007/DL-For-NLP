# install dependencies
# !pip install -U "transformers>=4.39.0"
# !pip install peft bitsandbytes
# !pip install -U "trl>=0.8.3"

import torch
import io
import PIL.Image as Image
from transformers import AutoTokenizer, AutoProcessor, TrainingArguments, LlavaForConditionalGeneration, BitsAndBytesConfig
from trl import SFTTrainer
from peft import LoraConfig
from datasets import Dataset
from huggingface_hub import notebook_login


def load_llava_model(model_id):
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    model = LlavaForConditionalGeneration.from_pretrained(model_id, quantization_config=quantization_config,
                                                          torch_dtype=torch.float16)
    torch.cuda.empty_cache()
    return model

class LLavaDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        texts = []
        images = []
        for example in examples:
            messages = example["messages"]
            text = self.processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
            img = Image.open(io.BytesIO(example['images'][0]['bytes']))
            images.append(img)

        batch = self.processor(texts, images, return_tensors="pt", padding=True)

        labels = batch["input_ids"].clone()
        if self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        return batch


def load_datasets(train_path, test_path):
    train_dataset = Dataset.load_from_disk(train_path)
    eval_dataset = Dataset.load_from_disk(test_path)
    return train_dataset, eval_dataset


def configure_training_args(output_dir="llava-1.5-7b-hf-ft-museum"):
    training_args = TrainingArguments(
        output_dir=output_dir,
        report_to="tensorboard",
        learning_rate=1.4e-5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        logging_steps=5,
        num_train_epochs=5,
        push_to_hub=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        fp16=True,
        bf16=False
    )
    return training_args


def configure_llora_config():
    lora_config = LoraConfig(
        r=32,
        lora_alpha=16,
        target_modules="all-linear"
    )
    return lora_config


def configure_trainer(model, args, train_dataset, eval_dataset, lora_config, tokenizer, data_collator):
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        dataset_text_field="text",  # need a dummy field
        tokenizer=tokenizer,
        data_collator=data_collator,
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    return trainer


def save_model(model, output_dir="llava_model_museum"):
    training_args = TrainingArguments(output_dir=output_dir, push_to_hub=False)
    model.save_pretrained(output_dir, push_to_hub=training_args.push_to_hub)


def main():
    # Load LLAVA model
    model_id = "llava-hf/llava-1.5-7b-hf"
    model = load_llava_model(model_id)

    # Load datasets
    train_path = '/kaggle/input/metmesuem-data/train.arrow'
    test_path = '/kaggle/input/metmesuem-data/test.arrow'
    train_dataset, eval_dataset = load_datasets(train_path, test_path)

    # Login to Hugging-face Hub
    notebook_login()

    # Configure training arguments
    args = configure_training_args()

    # Configure LLORA config
    lora_config = configure_llora_config()

    # Configure trainer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer = tokenizer
    data_collator = LLavaDataCollator(processor)
    trainer = configure_trainer(model, args, train_dataset, eval_dataset, lora_config, tokenizer, data_collator)

    # Load and set Tensorboard for logging
    # %load_ext tensorboard
    # %tensorboard --logdir llava-1.5-7b-hf-ft-museum

    # Train model
    trainer.train()

    # Save the Lora adapters to Hugging Face Hub
    trainer.push_to_hub()

    # Save model
    save_model(model)


if __name__ == "__main__":
    main()