from sklearn.base import BaseEstimator

def load_sent_word_net():
    """SentiWordNet 불러오기"""
    import csv, collections
    import numpy as np
    from io import StringIO
    from google.cloud import storage

    sent_scores = collections.defaultdict(list)

    storage_client = storage.Client()
    bucket = storage_client.get_bucket('twitter-bckt')
    blob = bucket.blob('SentiWordNet_3.0.0.txt')
    blob = blob.download_as_string()
    blob = blob.decode('utf-8')

    blob = StringIO(blob)  # tranform bytes to string here

    reader = csv.reader(blob, delimiter='\t', quotechar='"')  # then use csv library to read the content

    for line in reader:
        if line[0].startswith("#"):
            continue
        if len(line) == 1:
            continue
        POS, ID, PosScore, NegScore, SynsetTerms, Glos = line
        if len(POS) == 0 or len(ID) == 0:
            continue
        for term in SynsetTerms.split(" "):
            # print(term)
            # "# 뒤에 숫자 제거
            term = term.split('#')[0]
            # print(term)
            term = term.replace("-", " ").replace("_", " ")
            key = "%s/%s" % (POS, term)
            # print(key)
            sent_scores[key].append((float(PosScore), float(NegScore)))
            # print(sent_scores)
    for key, value in sent_scores.items():
        sent_scores[key] = np.mean(value, axis=0)
    print(key)

    return sent_scores

sent_word_net = load_sent_word_net()

# 사전 예시
# print(sent_word_net['v/fantasize'])

class LinguisticVectorizer(BaseEstimator):

    # def __init__(self):
    #     self._model = None
    #     self._project = 'engineering123'
    #     self._bucket_name = 'twitter-bckt'
    #     self._blob_name = 'SentiWordNet_3.0.0.txt'
    #     self._destination_name = 'SentiWordNet_3.0.0.txt'

    """POS 품사를 고려한 긍정 부정 단어 점수 사전 """

    def get_feature_names(self):
        import numpy as np
        # 중립을 고려하지 않음.
        return np.array(['sent_pos', 'sent_neg', 'nouns', 'adjectives', 'verbs', 'adverbs'])

    def fit(self, documents, y=None):
        return self

    def _get_sentiments(self, d):
        import nltk
        import numpy as np
        # print(d)
        print(d)
        print(d)
        print(type(d))
        sent = tuple(d.split())
        tagged = nltk.pos_tag(sent)
        # pos_tag와 SentiWordNet과 단어의 품사가 일치하지 않으면 긍정, 부정 점수는 모두 0

        pos_vals = []
        neg_vals = []

        nouns = 0.
        adjectives = 0.
        verbs = 0.
        adverbs = 0.

        i = 0
        for w, t in tagged:
            # 품사가 sentiword에 없는 경우 전체 개수 맞춰주기 위해 i 변수 설정

            p, n = 0, 0
            # print(w,t)
            sent_pos_type = None
            if t.startswith("NN"):
                # 명사
                sent_pos_type = "n"
                nouns += 1
            elif t.startswith("JJ"):
                # 형용사
                sent_pos_type = "a"
                adjectives += 1
            elif t.startswith("VB"):
                # 동사
                sent_pos_type = "v"
                verbs += 1
            elif t.startswith("RB"):
                # 부사
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
        import numpy as np
        pos_val, neg_val, nouns, adjectives, verbs, adverbs = np.array([self._get_sentiments(d) for d in documents]).T
        result = np.array([pos_val, neg_val, nouns, adjectives, verbs, adverbs]).T

        return result

if __name__ == '__main__':
    ling_stats = LinguisticVectorizer()