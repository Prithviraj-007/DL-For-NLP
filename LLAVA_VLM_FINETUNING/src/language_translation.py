
# !pip install googletrans==3.1.0a0
# !pip install transformers sentencepiece

from googletrans import Translator
from transformers import MarianMTModel, MarianTokenizer  # transformer based pre-trained language translation model
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast



def translate_hi2en_gtrans(sentence):
    """
    Function to translate from Hindi to English.

    Args:
    - sentence: string in Hindi

    Returns:
    - English translated text string

    """
    translator = Translator()
    output = translator.translate(sentence, dest='en', src='hi')
    return output.text

def translate_en2hi_gtrans(sentence):
    """
    Function to translate from English to Hindi.

    Args:
    - sentence: string in English

    Returns:
    - Hindi translated text string

    """
    translator = Translator()
    output = translator.translate(sentence, dest='hi', src='en')
    return output.text

# Translates text from source_lang to target_lang using the pre-trained model
def translate_en_hi_transformer(text):
    # Load the Pre-trained Model and Tokenizer for english to hindi
    model_name_en_hi = "Helsinki-NLP/opus-mt-en-hi"  # English to Hindi translation model
    tokenizer = MarianTokenizer.from_pretrained(model_name_en_hi)
    model_en_hi = MarianMTModel.from_pretrained(model_name_en_hi)
    encoded = tokenizer(text, return_tensors="pt")
    translated = model_en_hi.generate(**encoded)
    return tokenizer.batch_decode(translated, skip_special_tokens=True)[0]

# Translates text from Hindi to english using the pre-trained model
def translate_hi_en_transformer(text):
    # Load the Pre-trained Model and Tokenizer for hindi to english
    model_name_hi_en = "Helsinki-NLP/opus-mt-hi-en"  # Hindi to English translation model
    tokenizer_hi = MarianTokenizer.from_pretrained(model_name_hi_en)
    model_hi_en = MarianMTModel.from_pretrained(model_name_hi_en)
    encoded = tokenizer_hi(text, return_tensors="pt")
    translated = model_hi_en.generate(**encoded)
    return tokenizer_hi.batch_decode(translated, skip_special_tokens=True)[0]

def translate_mbart(text, source_lang, target_lang):
    # Load model and tokenizer outside the function
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name)

    # Set source language
    tokenizer.src_lang = source_lang
    # Encode the text
    encoded_text = tokenizer(text, return_tensors="pt")
    # Force target language token
    forced_bos_token_id = tokenizer.lang_code_to_id[target_lang]
    # Generate the translation
    generated_tokens = model.generate(**encoded_text, forced_bos_token_id=forced_bos_token_id)
    # Decode the translation
    translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return translation


if __name__ == "__main__":
    print(translate_hi2en_gtrans("मैं खुश हूँ!!!"))
    print(translate_en2hi_gtrans("I am happy!!!"))
    import pandas as pd

    # Read CSV file into a Pandas DataFrame
    df_en = pd.read_csv('Data_with_QnA.csv', usecols=['Question1', 'Answer1', 'Question2', 'Answer2', 'Question3', 'Answer3', 'Question4', 'Answer4'])
    df_en.head()

    # convert all the question answers from English to Hindi
    df_hi = df_en.applymap(translate_en2hi_gtrans)
    df_hi.head()

    # Save the modified DataFrame to a CSV file
    df_hi.to_csv('Hindi_QnA.csv', index=False)

    # English to Hindi example
    english_text = " What is the material used to create the chess set?"
    hindi_translation = translate_en_hi_transformer(english_text)
    print(f"English: {english_text}")
    print(f"Hindi: {hindi_translation}")

    # Hindi to English example
    hindi_text = "आपका दिन कैसा चल रहा है?"  # How is your day going?
    english_translation = translate_hi_en_transformer(hindi_text)
    print(f"Hindi: {hindi_text}")
    print(f"English: {english_translation}")

    # Example usage
    hindi_text = "हिन्दी साहित्य पर अगर समुचित परिप्रेक्ष्य में विचार किया जाए तो स्पष्ट होता है कि हिन्दी साहित्य का इतिहास अत्यन्त विस्तृत व प्राचीन है। सुप्रसिद्ध भाषा वैज्ञानिक डॉ० हरदेव बाहरी के शब्दों में, हिन्दी साहित्य का इतिहास वस्तुतः वैदिक काल से आरम्भ होता है। यह कहना ही ठीक होगा कि वैदिक भाषा ही हिन्दी है। इस भाषा का दुर्भाग्य रहा है कि युग-युग में इसका नाम परिवर्तित होता रहा है। कभी 'वैदिक', कभी 'संस्कृत', कभी 'प्राकृत', कभी'अपभ्रंश' और अब - हिन्दी।[1] आलोचक कह सकते हैं कि 'वैदिक संस्कृत' और 'हिन्दी' में तो जमीन-आसमान का अन्तर है। पर ध्यान देने योग्य है कि हिब्रू, रूसी, चीनी, जर्मन और तमिल आदि जिन भाषाओं को 'बहुत पुरानी' बताया जाता है, उनके भी प्राचीन और वर्तमान रूपों में जमीन-आसमान का अन्तर है; पर लोगों ने उन भाषाओं के नाम नहीं बदले और उनके परिवर्तित स्वरूपों को 'प्राचीन', 'मध्यकालीन', 'आधुनिक' आदि कहा गया, जबकि 'हिन्दी' के सन्दर्भ में प्रत्येक युग की भाषा का नया नाम रखा जाता रहा।"
    english_translation = translate_mbart(hindi_text, "hi_IN", "en_XX")
    print(english_translation) 

    english_text = "English literature, the body of written works produced in the English language by inhabitants of the British Isles (including Ireland) from the 7th century to the present day. The major literatures written in English outside the British Isles are treated separately under American literature, Australian literature, Canadian literature, and New Zealand literature. English literature has sometimes been stigmatized as insular. It can be argued that no single English novel attains the universality of the Russian writer Leo Tolstoy’s War and Peace or the French writer Gustave Flaubert’s Madame Bovary. Yet in the Middle Ages the Old English literature of the subjugated Saxons was leavened by the Latin and Anglo-Norman writings, eminently foreign in origin, in which the churchmen and the Norman conquerors expressed themselves. From this combination emerged a flexible and subtle linguistic instrument exploited by Geoffrey Chaucer and brought to supreme application by William Shakespeare. During the Renaissance the renewed interest in Classical learning and values had an important effect on English literature, as on all the arts; and ideas of Augustan literary propriety in the 18th century and reverence in the 19th century for a less specific, though still selectively viewed, Classical antiquity continued to shape the literature. All three of these impulses derived from a foreign source, namely the Mediterranean basin. The Decadents of the late 19th century and the Modernists of the early 20th looked to continental European individuals and movements for inspiration. Nor was attraction toward European intellectualism dead in the late 20th century, for by the mid-1980s the approach known as structuralism, a phenomenon predominantly French and German in origin, infused the very study of English literature itself in a host of published critical studies and university departments. Additional influence was exercised by deconstructionist analysis, based largely on the work of French philosopher Jacques Derrida."
    hindi_translation = translate_mbart(english_text, "en_XX", "hi_IN")
    print(hindi_translation) 

