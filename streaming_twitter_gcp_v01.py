import os

print(os.getcwd())

# runnning time track
from datetime import datetime

# Data Analysis
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns
# from wordcloud import WordCloud

# Data Preprocessing and Feature Engineering
import csv, collections
import re

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import nltk
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from sklearn.base import BaseEstimator
from sklearn.pipeline import FeatureUnion

# Model Selection and Validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer
ftwo_scorer = make_scorer(fbeta_score, beta=2)

# 알파벳이 아닌 문자
non_alphabet = re.compile('[^a-z]+')

def pre_processing(datframe):
    """데이터 전처리 작업"""
    df = datframe
    # #대문자 소문자로 변환
    df["tweet"] = df["tweet"].map(lambda x: x.lower())
    print(df.head())

    # 해쉬태그 또느 멘션 제거
    df["tweet"] = df["tweet"].str.replace(r'[\@\#]\S+', '')
    print(df.head())

    #HTML 제거
    df["tweet"] = df["tweet"].str.replace(r'<\w+[^>]+>', '')
    print(df.head())

    #email 제거
    df["tweet"] = df["tweet"].str.replace(r'(\w+\.)*\w+@(\w+\.)+[a-z]+', '')
    print(df.head())

    # http, https 제거
    #www[\.][^ ]+ url 뒤 띄어쓰기 안 될 경우 있는지 확인 후
    df["tweet"] = df["tweet"].str.replace(r'(http|ftp|https)://[-\w.]+(:\d+)?(/([\w/_.]*)?)?|www[\.]\S+', '')
    print(df.head())

    # 마침표 제거
    df["tweet"] = df["tweet"].str.replace(r'[\.]', '')

    # 불용어 제거 작업
    # 트위터 글 공백 구분자로 자른 후 불용어는 제거 후 리스트
    stop = stopwords.words('english')
    # 불용어 추가
    manual_sw_list = ['retweet', 'retwet', 'rt', 'oh', 'dm', 'mt', 'ht', 'ff', 'facebook', 'instagram', 'twitter', 'shoulda',  \
                      'woulda', 'coulda', 'might', 'im', 'tb', 'mysql', 'hah',"a", "an", "the", "and", "but", "if", "or", "because", \
                      "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", \
                      "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", \
                      "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", \
                      "most", "other", "some", "such",  "nor", "only", "own", "same", "so", "than", "too", "very", "s", "t", "just", "don", "now", 'twet']

    stop.extend(manual_sw_list)

    # print(stop)

    df['tweet'] = df['tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    print(df.head())

    # 표제어 추출. 동사 품사 기준으로 추출
    lem = WordNetLemmatizer()
    df['tweet'] = df['tweet'].apply(lambda x: ' '.join([lem.lemmatize(word, 'v') for word in x.split()]))
    print(df.head())

    # 구두점 제거 Normalization
    df["tweet"] = df["tweet"].str.replace(r'[^\w\s]', '')
    print(df.head())

    # 불필요한 숫자 제거
    df["tweet"] = df["tweet"].str.replace(r'[0-9]+', '')
    print(df.head())

    # 두 글자 이상 중복된 경우 알파벳 모두 두 글자로 만드는 작업
    df['tweet'] = df['tweet'].str.replace(r'([a-z])\1{1,}', r'\1')

    print(df.head())
    # 인코딩으로 인한 깨진 문자 제거
    df['tweet'] = df['tweet'].apply(
        lambda x: ' '.join([word for word in x.split() if non_alphabet.search(word) is None]))
    print(df.head())

    #haha 여러번 중복 되는 것 하나로 변경
    df['tweet'] = df['tweet'].str.replace(r'(ha)\1{1,}', r'\1')
    print(df.head())

    #
    # #문자가 없는 경우 제거
    # df['rm_empty'] = df['unique_haha'].apply(lambda x: ' '.join([word for word in x.split() if len(word) > 0]))
    # print(df.head())

    """
    예시 문장 참조    
    import re
    p = re.compile('[^a-z]+')
    m = p.search("trãªn")

    print(m)
    """

    return df['tweet']

def load_sent_word_net():
        """SentiWordNet 불러오기"""
        sent_scores = collections.defaultdict(list)

        with open(os.path.join(os.getcwd(), "SentiWordNet_3.0.0.txt"),"r") as csvfile:
            reader = csv.reader(csvfile, delimiter='\t', quotechar='"')

            for line in reader:
                if line[0].startswith("#"):
                    continue
                if len(line)==1:
                    continue
                POS, ID, PosScore, NegScore, SynsetTerms, Glos = line
                if len(POS)==0 or len(ID)==0:
                    continue
                for term in SynsetTerms.split(" "):
                    # print(term)
                    #"# 뒤에 숫자 제거
                    term = term.split('#')[0]
                    # print(term)
                    term = term.replace("-"," ").replace("_", " ")
                    key = "%s/%s"%(POS, term)
                    # print(key)
                    sent_scores[key].append((float(PosScore),float(NegScore)))
                    # print(sent_scores)
            for key, value in sent_scores.items():
                sent_scores[key] = np.mean(value, axis=0)

            return sent_scores

sent_word_net = load_sent_word_net()

#사전 예시
# print(sent_word_net['v/fantasize'])

class LinguisticVectorizer(BaseEstimator):
    """POS 품사를 고려한 긍정 부정 단어 점수 사전 """
    def get_feature_names(self):
        #중립을 고려하지 않음.
        return np.array(['sent_pos', 'sent_neg', 'nouns', 'adjectives', 'verbs', 'adverbs'])

    def fit(self, documents, y=None):
        return self

    def _get_sentiments(self, d):
        sent = tuple(d.split())
        tagged = nltk.pos_tag(sent)
        #pos_tag와 SentiWordNet과 단어의 품사가 일치하지 않으면 긍정, 부정 점수는 모두 0

        pos_vals = []
        neg_vals = []

        nouns = 0.
        adjectives = 0.
        verbs = 0.
        adverbs = 0.

        i = 0
        for w, t in tagged:
            #품사가 sentiword에 없는 경우 전체 개수 맞춰주기 위해 i 변수 설정

            p, n = 0, 0
            # print(w,t)
            sent_pos_type = None
            if t.startswith("NN"):
                #명사
                sent_pos_type = "n"
                nouns += 1
            elif t.startswith("JJ"):
                #형용사
                sent_pos_type = "a"
                adjectives += 1
            elif t.startswith("VB"):
                #동사
                sent_pos_type = "v"
                verbs += 1
            elif t.startswith("RB"):
                #부사
                sent_pos_type = "r"
                adverbs += 1
            else:
                sent_pos_type = "Nan"

                i += 1
                l = len(sent) - i

                if l == 0:
                    l = 1
                else:
                    pass

            if sent_pos_type is not None:


                sent_word = "%s/%s" % (sent_pos_type, w)

                if sent_word in sent_word_net:
                    # print(sent_word_net[sent_word])
                    p, n = sent_word_net[sent_word]
                elif sent_word == "Nan":
                    p, n = 0, 0

                pos_vals.append(p)
                neg_vals.append(n)

        if i == 0:
            l = len(sent)
        else:
            pass

        avg_pos_val = np.mean(pos_vals)
        avg_neg_val = np.mean(neg_vals)

        return [avg_pos_val, avg_neg_val, nouns / l, adjectives / l, verbs / l, adverbs / l]

    # print(_get_sentiments('This be fantastic'))

    def transform(self, documents):
        pos_val, neg_val, nouns, adjectives, verbs, adverbs = np.array([self._get_sentiments(d) for d in documents]).T
        result = np.array([pos_val, neg_val, nouns, adjectives, verbs, adverbs]).T

        return result

class Model():
    # 트위터 training data 불러오기
    df = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='latin', header=None,
                     names=['target', 'ids', 'date', 'flag', 'user', 'tweet'])

    # 불필요한 컬렁 삭제
    df = df.drop(["ids", "date", "flag", "user"], axis=1)

    # 긍정,부정 결과만 다루기 때문에 4를 1로 변경.
    df['target'] = df['target'].replace(4, 1)

    # 트위터 문자 데이터 전처리 완료 후 데이터프레임
    df['tweets'] = pre_processing(df)

    # 정규표현식으로 인해 공백이 된 데이터 제거
    df.drop(df[df["tweets"] == ''].index, inplace=True)
    df = df.reset_index(drop=True)
    """
    amx_feature 25 또는 30개 까지
    #트위터에 제한이 글자수 제한된 용량 만큼 쓰는 사람은 드물 것이다. 파라미터로 max_feature 설정하기
    df["word_count"] = df['tweet'].apply(lambda x: len(str(x).split()))
    sns.distplot(df.word_count, kde=False, rug=True)
    plt.show()
    
    df.drop("word_count", axis=1, inplace=True)
    
    """
    df["word_count"] = df['tweet'].apply(lambda x: len(str(x).split()))
    # sns.distplot(df.word_count, kde=False, rug=True)
    # plt.show()

    df.drop("word_count", axis=1, inplace=True)

    pos = df[df['target'] == 1]
    cloud = (' '.join(pos['tweet']))
    wcloud = WordCloud(width=1000, height=500).generate(cloud)
    plt.figure(figsize=(15, 5))
    plt.imshow(wcloud)
    plt.axis('off')
    plt.show()

    pos = df[df['target'] == 0]
    cloud = (' '.join(pos['tweet']))
    wcloud = WordCloud(width=1000, height=500).generate(cloud)
    plt.figure(figsize=(15, 5))
    plt.imshow(wcloud)
    plt.axis('off')
    plt.show()


    # pd.set_option("display.max_rows", None, "display.max_columns", None)
    print(df.count(axis=0))

    # 데이터 셔플
    text_train, text_test, y_train, y_test = train_test_split(df['tweets'], df['target'], random_state=0,
                                                              test_size=0.20)

    print(len(text_train))
    print(len(text_test))
    print(len(y_train))
    print(len(y_test))

    start_time = datetime.now()

    tfidf_ngrams = TfidfVectorizer(min_df=10)
    ling_stats = LinguisticVectorizer()

    all_features = FeatureUnion([('ling', ling_stats), ('tfidf', tfidf_ngrams)])

    clf = MultinomialNB()

    pipeline = Pipeline([('all', all_features), ('clf', clf)])

    param_grid = {
        'all__tfidf__max_features' : [25, 30],
        'all__tfidf__ngram_range': [(1,2),(1.3)],
        'clf__alpha': [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]}

    grid = GridSearchCV(pipeline
                        , param_grid
                        , n_jobs=-1
                        , cv=2
                        , return_train_score=True
                        , verbose=3)

    for param in grid.get_params().keys():
        print(param)

    grid.fit(text_train, y_train)

    results = pd.DataFrame.from_dict(grid.cv_results_)  # converting the results in to a dataframe
    results = results.sort_values(['param_alpha'])
    results.head()

    bestparam = grid.best_params_['alpha']  # extracting the best hyperparameter
    print("The best Alpha=", bestparam)
    bestparam = grid.best_params_['ngram_range']  # extracting the best hyperparameter
    print("The best ngram_range=", bestparam)
    bestparam = grid.best_params_['max_features']  # extracting the best hyperparameter
    print("The best max_features=", bestparam)

    print("best cross-validation score: {:.2f}".format(grid.best_score_))

    # x_test = pd.DataFrame({"tweet": ['I love trump']})
    # x_test['tweets'] = pre_processing(x_test)
    #
    # print(pre_processing(x_test))
    #
    # y_predicted = grid.predict(x_test['tweets'])
    # print(y_predicted)

if __name__ == '__main__':
    model = Model()

# print(grid.bset_estimator_)
#
# pipe = make_pipeline(TfidfVectorizer(),
#                      MultinomialNB())
# param_grid = {
#     'tfidfvectorizer__ngram_range': [(1, 1)],
#     'tfidfvectorizer__min_df': [3],
#     'tfidfvectorizer__use_idf': (True, False),
#     'tfidfvectorizer__norm': ('l1', 'l2'),
#     'multinomialnb__alpha': [0.01, 0.05, 0.1, 0.5, 1.0]}
#
# grid = GridSearchCV(pipe
#                    ,param_grid
#                    ,cv=5
#                    # ,n_jobs=3
#                    ,scoring=ftwo_scorer
#                    ,verbose=10)
#
# # 파라미터명 불러오기
# # for param in grid.get_params().keys():
# #     print(param)
#
# grid.fit(text_train, y_train)

# print("best cross-validation score: {:.2f}".format(grid.best_score_))
# print(grid.bset_estimator_)
#
# end_time = datetime.now()
#
# print('Duration: {}'.format(end_time - start_time))
#
# vectorizer = grid.best_estimator_.named_steps["tfidfvectorizer"]
#
# x_train = vectorizer.transform(text_train)
#
# max_value = x_train.max(axis=0).toarray().ravel()
# sorted_by_tfidf = max_value.argsort()
# feature_names = np.array(vectorizer.get_feature_names())
#
# print("Features with lowest tfidf:\n{}".format(
#     feature_names[sorted_by_tfidf[:150]]
# ))
# print("Features with highest tfidf:\n{}".format(
#     feature_names[sorted_by_tfidf[-150:]]
# ))
#
# pipeline = Pipeline([
#     ('bow',CountVectorizer(analyzer=text_processing)),  # strings to token integer counts
#     ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
#     ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
# ])

# 중복 되는 단어 제거 방안 찾기.
# 예: aaaaawsome, aaaaaamazing

# tf-idf 사용.

#
# pipeline = Pipeline([
#     ('bow',CountVectorizer(analyzer=text_processing)),  # strings to token integer counts
#     ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
#     ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
# ])


# # 데이터 탐색
# col ='target'
#
# fig, ax = plt.subplots(figsize=(8, 5))
# sns.countplot(x=col, data=df, alpha=0.5)
#
# plt.show()


# word_features = get_word_features(get_words_in_tweets(tweets))


# for col in df.columns:
#
#     print(col)
#
#     # 전체 데이터 개수
#     print(len(df))
#
#     # 데이터 탐색
#     fig, ax = plt.subplots(figsize=(8, 5))
#     sns.countplot(x=col, data=df, alpha=0.5)
#
#     plt.show()



