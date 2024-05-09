
# !pip install -q -U google-generativeai

import time
import json
import pandas as pd
from google.colab import userdata
import google.generativeai as genai

def configure_genai():
    # Used to securely store your API key
    GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
    genai.configure(api_key=GOOGLE_API_KEY)

def generate_art_questions(data_json, model):
    art_texts = []
    i = 0

    for data_entry in data_json:
        # Prompt for generating questions based on art description and metadata
        prompt = f"Following is the description and meta data of an art. Based on this generate 4 questions and corresponding answer. Denote each question as **Question and answer as **Answer. {data_entry}"
        # Generate content using the model
        response = model.generate_content(prompt)
        art_texts.append(response.text)
        i += 1
        if i == 60:
            print("60 completed")
            i = 0
            time.sleep(10)
    
    return art_texts

def extract_qa_pairs(text):
    qa_pairs = {}
    current_q = 1
    current_a = 1

    for line in text.splitlines():
        if line.startswith("**Question"):
            qa_pairs[f"Question{current_q}"] = line.strip().split(":", 1)[1].strip("**")  # Extract question
            current_q += 1
        elif line.startswith(("**Answer", "Answer")):
            qa_pairs[f"Answer{current_a}"] = line.strip().split(":", 1)[1].strip("**")  # Extract answer
            current_a += 1

    return qa_pairs

def create_qa_dataframe(art_texts):
    df = pd.DataFrame(columns=["Question1", "Answer1", "Question2", "Answer2", "Question3", "Answer3", "Question4", "Answer4"])

    for art_text in art_texts:
        qa_dict = extract_qa_pairs(art_text)

        if len(list(qa_dict.values())) == 8:
            # Fill the DataFrame with extracted data
            df.loc[len(df)] = list(qa_dict.values())  # Efficiently fill with all values
        else:
            df.loc[len(df)] = [None]*8
    
    return df

def main():
    # Configure GenAI
    configure_genai()

    # Initialize Generative Model
    model = genai.GenerativeModel('gemini-pro')

    # Read the data from CSV
    data = pd.read_csv('/content/Data_for_questioning.csv')

    # Convert DataFrame to JSON
    data.to_json('final_data_json.json', orient='records', indent=4)

    # Open the JSON file and load its contents
    with open('/content/final_data_json.json', 'r') as f:
        data_json = json.load(f)

    # Generate questions for art descriptions
    art_texts = generate_art_questions(data_json, model)

    # Create DataFrame with questions and answers
    df = create_qa_dataframe(art_texts)

    # Save DataFrame to CSV
    df.to_csv('QnA.csv', index=False)

if __name__ == "__main__":
    main()
