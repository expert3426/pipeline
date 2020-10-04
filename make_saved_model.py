import os

print(os.getcwd())

# runnning time track
from datetime import datetime
import pickle

# Data Analysis
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

# Data Preprocessing and Feature Engineering
import csv, collections
import re
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import nltk

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from sklearn.base import BaseEstimator
from sklearn.pipeline import FeatureUnion
import sklearn
import joblib

# Model Selection and Validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score, make_scorer

import ling_stats

from sklearn.metrics import precision_recall_curve

filename = 'bayse_model.sav'

# 알파벳이 아닌 문자
non_alphabet = re.compile(r'[^a-z]+')

emo_info = {
    # positive emoticons
    ":‑)": " good ",
    ":)": " good ",
    ";)": " good ",
    ":-}": " good ",
    "=]": " good ",
    "=)": " good ",
    ":d": " good ",
    ":dd": " good ",
    "xd": " good ",
    "<3": " love ",

    ":p": "playful",
    "xp": "playful",

    # negativve emoticons
    ":‑(": " sad ",
    ":‑[": " sad ",
    ":(": " sad ",
    "=(": " sad ",
    "=/": " sad ",
    ":{": " sad ",
    ":/": " sad ",
    ":|": " sad ",
    ":-/": " sad ",
    ":o": " shock "

}
# 순서 지정 ":dd"가 ":d"보다 먼저 변환되기 때문.
emo_info_order = [k for (k_len, k) in reversed(sorted([(len(k), k) for k in emo_info.keys()]))]


def emo_repl(phrase):
    for k in emo_info_order:
        phrase = phrase.replace(k, emo_info[k])
    return phrase

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"\bdon't\b", "do not", phrase)
    phrase = re.sub(r"\bdoesn't\b", "does not", phrase)
    phrase = re.sub(r"\bdidn't\b", "did not", phrase)
    phrase = re.sub(r"\bdidnt\b", "did not", phrase)
    phrase = re.sub(r"\bhasn't\b", "has not", phrase)
    phrase = re.sub(r"\bhaven't\b", "have not", phrase)
    phrase = re.sub(r"\bhavent\b", "have not", phrase)
    phrase = re.sub(r"\bhadn't\b", "had not", phrase)
    phrase = re.sub(r"\bwon't\b", "will not", phrase)
    phrase = re.sub(r"\bwouldn't\b", "would not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)  # 소유격일 경우 처리 불가능..

    # using regular expressions to expand the contractions
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    # phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)

    return phrase


def pre_processing(datframe):
    """데이터 전처리 작업"""
    df = datframe
    # #대문자 소문자로 변환
    df["tweet"] = df["tweet"].map(lambda x: x.lower())
    print(df.head())

    # email 제거
    df["tweet"] = df["tweet"].str.replace(r'(\w+\.)*\w+@(\w+\.)+[a-z]+', '')
    print(df.head())

    # http, https 제거
    # www[\.][^ ]+ url 뒤 띄어쓰기 안 될 경우 있는지 확인 후
    df["tweet"] = df["tweet"].str.replace(r'(http|ftp|https)://[-\w.]+(:\d+)?(/([\w/_.]*)?)?|www[\.]\S+', '')
    print(df.head())

    # 해쉬태그 또느 멘션 제거
    df["tweet"] = df["tweet"].str.replace(r'[\@\#]\S+', '')
    print(df.head())

    # HTML 관련 문자 제거
    df["tweet"] = df["tweet"].str.replace(r'<\w+[^>]+>|[\&]\w+[\;]', '')
    print(df.head())

    # 마침표 제거
    df["tweet"] = df["tweet"].str.replace(r'[\.]', '')

    # 이모티콘 문자 변환
    df['tweet'] = df['tweet'].apply(emo_repl)
    print(df.head())

    #"-" 또는 "_"로 묶여 있는 단어 분리(예시: never-ending => never ending)
    df["tweet"] = df["tweet"].str.replace(r'[\-\_]', ' ')

    # 발음 수축된 단어 두 단어로 다시 표기(예시: i'm => i am
    df['tweet'] = df['tweet'].apply(decontracted)
    print(df.head())

    # 불용어 제거 작업
    # 트위터 글 공백 구분자로 자른 후 불용어는 제거 후 리스트
    stop = stopwords.words('english')
    # 불용어 추가
    manual_sw_list = ['retweet', 'retwet', 'rt', 'oh', 'dm', 'mt', 'ht', 'ff', 'shoulda' \
                      'woulda', 'coulda', 'might', 'im', 'tb', 'mysql', 'hah', "a", "an", "the", "and", "but", "if",
                      "or", "because", \
                      "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
                      "through", "during", \
                      "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over",
                      "under", "again", \
                      "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both",
                      "each", "few", "more", \
                      "most", "other", "some", "such", "nor", "only", "own", "same", "so", "than", "too", "very", "s",
                      "t", "just", "don", "now", 'tweet', 'x', 'f','']

    stop.extend(manual_sw_list)

    # print(stop)
    df['tweet'] = df['tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    print(df.head())

    # 표제어 추출. 동사 품사 기준으로 추출
    lem = WordNetLemmatizer()
    df['tweet'] = df['tweet'].apply(lambda x: ' '.join([lem.lemmatize(word, 'v') for word in x.split()]))
    print(df.head())

    # 구두점 제거
    df["tweet"] = df["tweet"].str.replace(r'[^\w\s]', '')
    print(df.head())

    # 불필요한 숫자 제거
    df["tweet"] = df["tweet"].str.replace(r'[0-9]+', '')
    print(df.head())

    print(df.head())
    # 인코딩으로 인한 깨진 문자 제거
    df['tweet'] = df['tweet'].apply(lambda x: ' '.join([word for word in x.split() if non_alphabet.search(word) is None]))
    print(df.head())

    # haha 여러번 중복 되는 것 두글자로 변경
    df['tweet'] = df['tweet'].str.replace(r'(ha)\1{1,}', r'\1')
    print(df.head())

    # 두 글자 이상 중복된 경우 알파벳 모두 두 글자로 만드는 작업(예시: woooow -> woow
    df['tweet'] = df['tweet'].str.replace(r'([a-z])\1{1,}', r'\1\1')
    print(df.head())

    #두 글자 이상 중복되는 알파벳일 경우, 영어사전에 없는 단어는 한 글자로 줄이기(예시: woow, 사전에 존재 하지 않음 wow로 변경
    df['tweet'] = df['tweet'].apply(lambda x: ' '.join([word if len(wordnet.synsets(word)) > 0 else re.sub(r'([a-z])\1{1,}', r'\1', word) for word in x.split()]))
    print(df.head())

    # #문자가 없는 경우 제거
    # df['rm_empty'] = df['unique_haha'].apply(lambda x: ' '.join([word for word in x.split() if len(word) > 0]))
    # print(df.head())


    return df['tweet']

class Model():

    # 트위터 training data 불러오기
    df = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='latin', header=None,
                     names=['target', 'ids', 'date', 'flag', 'user', 'tweet'])

    # 불필요한 컬렁 삭제
    df = df.drop(["ids", "date", "flag", "user"], axis=1)

    # 긍정,부정 결과만 다루기 때문에 4를 1로 변경.(설명변수)
    df['target'] = df['target'].replace(4, 1)

    # 트위터 문자 데이터 전처리 완료 후 데이터프레임(독립변수)
    df['tweets'] = pre_processing(df)

    # 정규표현식으로 인해 공백이 된 데이터 제거
    df.drop(df[df["tweets"] == ''].index, inplace=True)
    df = df.reset_index(drop=True)

    # pd.set_option("display.max_rows", None, "display.max_columns", None)
    print(df.count(axis=0))

    # 데이터 셔플
    # 분석용, 검증 데이터 분리
    text_train, text_validation, y_train, y_validation = train_test_split(df['tweets'], df['target'], random_state=0, test_size=0.20)

    print(len(text_train))
    print(len(text_validation))
    print(len(y_train))
    print(len(y_validation))

    # 테스트 데이터로 모델 평가
    df_test = pd.read_csv('trump_labeled_final.csv', header=None, names=['tweet', 'target'])

    # 트위터 문자 데이터 전처리 완료 후 데이터프레임(독립변수)
    df_test['tweets'] = pre_processing(df_test)
    # 정규표현식으로 인해 공백이 된 데이터 제거
    df_test.drop(df_test[df_test["tweets"] == ''].index, inplace=True)
    df_test = df_test.reset_index(drop=True)

    text_test = df_test['tweets']
    y_test = df_test['target']

    start_time = datetime.now()

    # Modelling

    tfidf_ngrams = TfidfVectorizer(min_df=5, ngram_range=(1, 3))
    ling_stats = ling_stats.LinguisticVectorizer()
    all_features = FeatureUnion([('ling', ling_stats), ('tfidf', tfidf_ngrams)])
    clf = MultinomialNB(alpha=5)

    pipeline = Pipeline([('all', all_features), ('clf', clf)])

    pipeline.fit(text_train, y_train)

    joblib_file = "bayse_model.pkl"
    joblib.dump(pipeline, joblib_file)


if __name__ == '__main__':
    model = Model()
