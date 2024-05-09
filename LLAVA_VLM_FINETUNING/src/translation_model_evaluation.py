import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import corpus_bleu
from indicnlp.tokenize import indic_tokenize
import pandas as pd
import numpy as np

def configure_nltk():
    nltk.download('punkt')
    nltk.download('wordnet')

def tokenize_hindi(text):
    return indic_tokenize.trivial_tokenize(text, lang='hi')

def calculate_meteor_score(ref, output):
    ref_tokens = tokenize_hindi(ref)
    output_tokens = tokenize_hindi(output)
    return meteor_score([ref_tokens], output_tokens)

def calculate_bleu_score(ref, output):
    ref_tokens = [tokenize_hindi(ref)]
    output_tokens = tokenize_hindi(output)
    return corpus_bleu(ref_tokens, [output_tokens])

def read_hindi_data(filepath):
    hindi_df = pd.read_csv(filepath)
    hindi_df = hindi_df.replace('नेन', np.nan)
    hindi_df = hindi_df.dropna()
    return hindi_df

def create_reference_df(hindi_df):
    reference_df = pd.DataFrame()
    for i in range(1, 3):
        reference_df[f'Question{i}'] = hindi_df[f'Question{i}']
        reference_df[f'Answer{i}'] = hindi_df[f'Answer{i}']
    return reference_df

def select_first_n_rows(df, n=10):
    return df.iloc[:n]

def calculate_scores(reference_df, n=10):
    meteor_scores = []
    bleu_scores = []

    for i in range(1, 3):  
        ref_question_col = f'Question{i}'
        ref_answer_col = f'Answer{i}'
        output_question_col = f'Question{i}_hindi'
        output_answer_col = f'Answer{i}_hindi'

        for ref_question, output_question in zip(reference_df[ref_question_col], reference_df[output_question_col]):
            meteor_scores.append(calculate_meteor_score(ref_question, output_question))
            bleu_scores.append(calculate_bleu_score(ref_question, output_question))

        for ref_answer, output_answer in zip(reference_df[ref_answer_col], reference_df[output_answer_col]):
            meteor_scores.append(calculate_meteor_score(ref_answer, output_answer))
            bleu_scores.append(calculate_bleu_score(ref_answer, output_answer))

    average_meteor_score = sum(meteor_scores) / len(meteor_scores)
    average_bleu_score = sum(bleu_scores) / len(bleu_scores)
    
    return average_meteor_score, average_bleu_score

def main():
    # Configure NLTK
    configure_nltk()

    # Read Hindi data
    hindi_df = read_hindi_data('/content/Hindi_QnA.csv')

    # Create reference dataframe
    reference_df = create_reference_df(hindi_df)

    # Select first 10 rows
    reference_df_first10 = select_first_n_rows(reference_df)

    # Calculate scores
    average_meteor_score, average_bleu_score = calculate_scores(reference_df_first10)

    print("Average METEOR Score:", average_meteor_score)
    print("Average BLEU Score:", average_bleu_score)

if __name__ == "__main__":
    main()
