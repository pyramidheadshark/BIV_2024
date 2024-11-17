import os
import re
import pathlib

import nltk

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer
from wordcloud import WordCloud
from PIL import Image
from ydata_profiling import ProfileReport

def clean_text_from_specsymbols(text):
    return re.sub(r"[^a-zA-Zа-яА-ЯёЁ\s]", "", text)

def clean_text_from_single_letters(text):
    return re.sub(r"\b[а-яёa-zA-Z0-9]\b", "", text, flags=re.IGNORECASE)

def clean_text_from_double_letters(text):
    return re.sub(r"\b[a-zA-Zа-яА-Я]{2}\b", "", text, flags=re.IGNORECASE)

def clean_text_from_unneccesary(text):
    words = nltk.word_tokenize(text)
    
    russian_stopwords = stopwords.words('russian')

    clean_words = []
    for word in words:
        lemma = morph.parse(word)[0].normal_form
        if lemma not in russian_stopwords:
            clean_words.append(lemma)

    return ' '.join(clean_words)
        
if __name__ == "__main__":
    # data_path = os.environ['DATA_PATH']
    current_dir = pathlib.Path.cwd()
    data_dir = current_dir / 'data'
    
    predictions_file = data_dir / 'predictions.tsv'
    payments_file = data_dir / 'payments_main.tsv'
     
    nltk.download('stopwords')
    morph = MorphAnalyzer()
    
    df_pred = pd.read_csv(predictions_file, sep='\t', index_col=0)
    df_orig = pd.read_csv(payments_file, sep='\t', index_col=0, names=['Date', 'Amount', 'Description'])

    df_orig['Category'] = df_pred
    