import time
import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import coo_matrix, hstack
import pickle
from spellchecker import SpellChecker
import textstat
from sklearn.preprocessing import MinMaxScaler 

import helpers.utils as utils
# Load the essays from a CSV file
train = pd.read_csv('data/nzqa/Training/train_S2Q2.csv')
train = utils.preprocess_data(train, 'Q2')
train, train_zeros = utils.process_zeros(train, 'Q2')
#train = train.head(1)  # Assuming we are only using the first row for testing

def preprocess_paragraph(text):
    # Replace occurrences of double newlines or newline characters with a single \n for paragraph breaks
    text = text.replace('\n\n', '\n').replace('\r\n', '\n').replace('\r', '\n')
    return text

# Apply the preprocessing to the entire 'Response' column
train['Response'] = train['Response'].apply(preprocess_paragraph)
print(train["Response"])

# Split paragraphs
train['paragraph'] = train['Response'].str.split('\n')
print(train["paragraph"])

def remove__(x):
    """Removes multiple spaces"""
    return re.sub("\s{2,100000}", "", x)

def clean_row_for_split(row):
    """Preprocessing required to split the words"""
    row = row.lower()
    row = re.sub("[<>\|\^\@\*²¹©\$\d\&,\/\.\(\)\[\]\s\&\%\#\!\-\_\=\+\:\?\"';]"," ", row)
    row = re.sub("\s{2,10000}", " ", row)
    row = row.split(" ")
    return row

spell = SpellChecker()
def Spelling_Errors(row):
    num = 0
    nn = len(row)
    for tok in row:
        if spell.unknown([tok]):
            num += 1
    return num / nn

def Unique_words_prcnt(row):
    return len(set(row)) / len(row)

# Since we have already split the text as paragraphs, now we will remove all unnecessary spaces
train['Response'] = train['Response'].apply(remove__)

def PreprocessText(x):
    x = x.lower()
    x = re.sub("[0-9\&\,\.\_\~\+\-\(\)\*#\$\%@!\?:;`\|\"'\[\]\/\\\]", " ", x)
    x = re.sub("[\s\.]{2,100000}", " ", x)
    return x

def ParaPreprocessing(train):
    train = train.explode('paragraph')
    train['p_cln'] = train['paragraph'].apply(clean_row_for_split)
    train['paragraph_len'] = train['paragraph'].apply(len)
    train['paragraph_sent_cnt'] = train['paragraph'].apply(lambda x: len(x.split('.')))
    train['paragraph_word_cnt'] = train['paragraph'].apply(lambda x: len(x.split(' ')))
    train['paragraph_errors'] = train['p_cln'].apply(Spelling_Errors)
    return train

# Feature engineering
paragraph_fea = ['paragraph_len', 'paragraph_sent_cnt', 'paragraph_word_cnt', 'paragraph_errors']

def ParaFeatures(train_tmp):
    aggs = {
        'paragraph_len': ['max', 'mean', 'min', 'sum', 'first', 'last', 'quantile'],
        'paragraph_sent_cnt': ['max', 'mean', 'min', 'sum', 'first', 'last', 'quantile'],
        'paragraph_word_cnt': ['max', 'mean', 'min', 'sum', 'first', 'last', 'quantile'],
        'paragraph_errors': ['max', 'mean', 'min', 'sum', 'first', 'last', 'quantile']
    }
    agg_df = train_tmp.groupby('Unique_ID').agg(aggs)
    agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
    return agg_df.reset_index()

def SentPreprocessing(train):
    train['sentence'] = train['Response'].str.split('.')
    train = train.explode('sentence')
    train['s_cln'] = train['sentence'].apply(clean_row_for_split)
    train['sentence_len'] = train['sentence'].apply(len)
    train['sentence_word_cnt'] = train['sentence'].apply(lambda x: len(x.split(' ')))
    train['sentence_errors'] = train['s_cln'].apply(Spelling_Errors)
    train['s_unique_words_prcnt'] = train['s_cln'].apply(Unique_words_prcnt)
    return train

sent_feats = ["sentence_len", "sentence_word_cnt", "sentence_errors", "s_unique_words_prcnt"]

def SentFeatures(train):
    aggs = {
        'sentence_len': ['max', 'mean', 'min', 'sum', 'first', 'last', 'quantile'],
        'sentence_word_cnt': ['max', 'mean', 'min', 'sum', 'first', 'last', 'quantile'],
        'sentence_errors': ['max', 'mean', 'min', 'sum', 'first', 'last', 'quantile'],
        's_unique_words_prcnt': ['max', 'mean', 'min', 'sum','first', 'last', 'quantile']
    }
    agg_df = train.groupby('Unique_ID').agg(aggs)
    agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
    return agg_df.reset_index()

def WordPreprocessing(train):
    train['word'] = train['Response'].str.split(' ')
    train = train.explode('word')
    train['word_len'] = train['word'].apply(len)
    train = train[train['word_len'] > 0]
    return train

def WordFeatures(train):
    aggs = {
        'word_len': ['max', 'mean', 'min', 'quantile']
    }
    agg_df = train.groupby('Unique_ID').agg(aggs)
    agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
    return agg_df.reset_index()

# Paragraph Features
para_df = ParaPreprocessing(train)
para_df = ParaFeatures(para_df)

# Sentence Features
sent_df = SentPreprocessing(train)
sent_df = SentFeatures(sent_df)

# Word Features
word_df = WordPreprocessing(train)
word_df = WordFeatures(word_df)

#merge together and save to csv
def get_feats(para=para_df, sent=sent_df, word=word_df):
    """Returns all newly created features"""
    feats = pd.merge(para, sent, how='inner', on='Unique_ID')
    feats = pd.merge(feats, word, how='inner', on='Unique_ID')
    return feats

feats = get_feats()

#readbility features
def calculate_fk_score(txt):
    return textstat.flesch_kincaid_grade(txt)

def calculate_fr_score(txt):
    return textstat.flesch_reading_ease(txt)

def calculate_dc_score(txt):
    return textstat.dale_chall_readability_score(txt)

def calculate_ar_score(txt):
    return textstat.automated_readability_index(txt)

def calculate_dw_score(txt):
    return textstat.difficult_words(txt)

def get_readability_features(df):
    df['fk_score'] = df['Response'].apply(calculate_fk_score)
    df['ar_score'] = df['Response'].apply(calculate_ar_score)
    df['fr_score'] = df['Response'].apply(calculate_fr_score)
    df['dc_score'] = df['Response'].apply(calculate_dc_score)
    df['dw_score'] = df['Response'].apply(calculate_dw_score)
    return df

train = get_readability_features(train)
manual_features = pd.merge(feats, train[['Unique_ID', 'fk_score', 'ar_score', 'fr_score', 'dc_score', 'dw_score']], on='Unique_ID', how='inner')
#nomarlisze the features
scaler = MinMaxScaler()
manual_features.iloc[:, 1:] = scaler.fit_transform(manual_features.iloc[:, 1:])


manual_features.to_csv("data/nzqa/Training/train_manual_features_Q2.csv", index=False)
print(manual_features.head())