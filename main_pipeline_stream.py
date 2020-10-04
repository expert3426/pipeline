from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions, StandardOptions
from google.cloud import pubsub_v1
from google.cloud import bigquery
from google.cloud import storage
import apache_beam as beam
import logging
import argparse

PROJECT = 'engineering123'
SCHEMA = 'tweet:STRING, proc_tweet:STRING ,user_id:STRING, tweet_tp:STRING, exp_score:NUMERIC, loc:STRING, reg_dt:TIMESTAMP'
TOPIC = "projects/engineering123/topics/twitter_subject"

trump_cnt = re.compile(r'\btrump\b')
biden_cnt = re.compile(r'\bbiden\b')
# 알파벳이 아닌 문자
non_alphabet = re.compile('[^a-z]+')

class pre_processing(beam.DoFn):
    def process(self, element):

        # 두 글자가 중복되는 문자
        import re
        dupl_wd = re.compile(r'([a-z])\1{1,}')
        import nltk
        from nltk.stem.wordnet import WordNetLemmatizer as wnt

        nltk.download('wordnet')
        nltk.download('averaged_perceptron_tagger')
        from nltk.corpus import wordnet
        # wordnet.ensure_loaded()

        from nltk.corpus import stopwords
        import base64
        import pandas as pd
        non_alphabet = re.compile('[^a-z]+')
        # 두 글자가 중복되는 문자

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
            ":p": "good",
            "xp": "good",
            "<3": " love ",

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
            import re
            import json
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

        tweets = []
        stream = base64.urlsafe_b64decode(element)
        twraw = json.loads(stream)
        twmessages = twraw.get('messages')
        for message in twmessages:
            tweets.append(message['data'])

        newlist = []

        for k, v in tweets[0].items():
            if k == 'text':
                newlist.append(v)
            elif k == 'user_id':
                newlist.append(v)
            elif k == 'location':
                newlist.append(v)
            elif k == 'date':
                newlist.append(v)
            else:
                pass

        df = pd.DataFrame(newlist)
        df = df.T
        df.rename(columns={0: 'date', 1: 'tweet', 2: 'user_id', 3: 'location'}, inplace=True)

        df['raw'] = df['tweet']
        print(df)

        """데이터 전처리 작업"""
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

        # "-" 또는 "_"로 묶여 있는 단어 분리(예시: never-ending => never ending)
        df["tweet"] = df["tweet"].str.replace(r'[\-\_]', ' ')

        # 발음 수축된 단어 두 단어로 다시 표기(예시: i'm => i am
        df['tweet'] = df['tweet'].apply(decontracted)
        print(df.head())

        # 불용어 제거 작업
        # 트위터 글 공백 구분자로 자른 후 불용어는 제거 후 리스트
        stop = stopwords.words('english')
        # 불용어 추가
        manual_sw_list = ['retweet', 'retwet', 'rt', 'oh', 'dm', 'mt', 'ht', 'ff', 'shoulda' \
                                                                                   'woulda', 'coulda', 'might', 'im',
                          'tb', 'mysql', 'hah', "a", "an", "the", "and", "but", "if",
                          "or", "because", \
                          "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between",
                          "into",
                          "through", "during", \
                          "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off",
                          "over",
                          "under", "again", \
                          "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any",
                          "both",
                          "each", "few", "more", \
                          "most", "other", "some", "such", "nor", "only", "own", "same", "so", "than", "too", "very",
                          "s",
                          "t", "just", "don", "now", 'tweet', 'x', 'f', '']

        stop.extend(manual_sw_list)

        # print(stop)
        df['tweet'] = df['tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
        print(df.head())

        # 표제어 추출. 동사 품사 기준으로 추출
        lem = wnt()
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
        df['tweet'] = df['tweet'].apply(
            lambda x: ' '.join([word for word in x.split() if non_alphabet.search(word) is None]))
        print(df.head())

        # haha 여러번 중복 되는 것 두글자로 변경
        df['tweet'] = df['tweet'].str.replace(r'(ha)\1{1,}', r'\1')
        print(df.head())

        # 두 글자 이상 중복된 경우 알파벳 모두 두 글자로 만드는 작업(예시: woooow -> woow
        df['tweet'] = df['tweet'].str.replace(r'([a-z])\1{1,}', r'\1\1')
        print(df.head())

        # 두 글자 이상 중복되는 알파벳일 경우, 영어사전에 없는 단어는 한 글자로 줄이기(예시: woow, 사전에 존재 하지 않음 wow로 변경
        df['tweet'] = df['tweet'].apply(lambda x: ' '.join([word if len(wordnet.synsets(word)) > 0 else re.sub(r'([a-z])\1{1,}', r'\1', word) for word in x.split()]))
        print(df.head())

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

        return [df]

"""데이터 전처리 부분"""
class discard_incomplete(beam.DoFn):

  def process(self, element):
      import re

      trump_cnt = re.compile(r'\btrump\b')
      biden_cnt = re.compile(r'\bbiden\b')
      # 알파벳이 아닌 문자
      # 조건에 일치하는 데이터프레임만 반환하기
      # 트위터 전처리 후 nuill 값인 경우, user_id 없는 경우, trump,biden 모두 있는 경우와 없는 경우 제외
      if len(element['tweet'][0]) > 0 and \
             len(element['user_id'][0]) > 0 and \
             (trump_cnt.search(element['tweet'][0]) is not None or biden_cnt.search(element['tweet'][0]) is not None) and \
             not (trump_cnt.search(element['tweet'][0]) is not None and biden_cnt.search(element['tweet'][0]) is not None):
          yield element
      else:
          return

class PredictSklearn(beam.DoFn):
    """pardo return value는 반드시 list"""
    """ Format the input to the desired shape"""

    def __init__(self):
        self._model = None
        self._project = 'engineering123'
        self._bucket_name = 'twitter-bckt'
        self._blob_name = 'bayse_model.pkl'
        self._destination_name = 'bayse_model.pkl'

    # #모델 spin 작업.
    def setup(self):

        # dataflow 버전 temp 파일에서 작업하기
        import joblib
        from google.cloud import storage
        from tempfile import TemporaryFile

        logging.info(
            "Sklearn model initialisation {}".format(self._blob_name))

        storage_client = storage.Client()
        bucket = storage_client.get_bucket(self._bucket_name)
        blob = bucket.blob(self._blob_name)

        with TemporaryFile() as temp_file:
            # download blob into temp file
            blob.download_to_file(temp_file)
            temp_file.seek(0)
            # load into joblib
            self._model = joblib.load(temp_file)

    def process(self, element):

        import re
        import datetime

        raw_twt = element['raw'][0]
        twt = element['tweet'][0]
        print(type(twt))
        trump_cnt = re.compile(r'\btrump\b')

        if trump_cnt.search(twt) is not None:
            tp = 'TRP'
            pred_value = self._model.predict([twt])
            pred_value = pred_value.item()
        else:
            tp = 'BID'
            pred_value = self._model.predict([twt])
            pred_value = pred_value.item()

        rst = {
            'tweet': raw_twt,
            'proc_tweet': twt,
            'user_id': element["user_id"][0],
            'tweet_tp': tp,
            'exp_score': pred_value,
            'loc' : element["location"][0],
            'reg_dt': element["date"][0],
        }

        return [rst]


def main(argv=None):
    import apache_beam.transforms.window as window
    import apache_beam.transforms.trigger as trigger

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_topic',
                        default='projects/engineering123/topics/twitter_subject',
                        required=True,
                        help='Topic to pull data from.')

    known_args, pipeline_args = parser.parse_known_args(argv)

    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = True
    pipeline_options.view_as(StandardOptions).streaming = True

    p = beam.Pipeline(options=PipelineOptions())

    (p
     | 'ReadData' >> beam.io.ReadFromPubSub(topic=known_args.input_topic)
     | "Clean Data" >> beam.ParDo(pre_processing())
     | 'DelIncompleteData' >> beam.ParDo(discard_incomplete())  # return 값 false는 무시한다.
     | 'predict label' >> beam.ParDo(PredictSklearn())
     | 'WriteToBigQuery' >> beam.io.WriteToBigQuery('{0}:twitter_project.sent_anl_rst'.format(PROJECT), schema=SCHEMA,
                                                    write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND)
     )
    result = p.run()
    result.wait_until_finish()

if __name__ == '__main__':
    logger = logging.getLogger().setLevel(logging.INFO)
    main()