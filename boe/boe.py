# https://blog.manash.me/how-to-use-pre-trained-word-vectors-from-facebooks-fasttext-a71e6d55f27
# http://ai.intelligentonlinetools.com/ml/fasttext-word-embeddings-text-classification-python-mlp/


import numpy as np
import spacy
from spacy.tokenizer import Tokenizer
import dateutil.parser as parser
import datetime

EMB_DIM = 300

# WOTD events range between 1300-today, so only 22 numbers are
# needed to encode thousands, hundreds and tens
YEAR_ENCODING_DIM = 22

nlp = spacy.load('en_core_web_lg')

# stop words missing for en_core_web_lg for spaCy v2.0, see https://github.com/explosion/spaCy/issues/1574
# A workaround:
for word in nlp.Defaults.stop_words:
    for w in (word, word[0].upper() + word[1:], word.upper()):
        lex = nlp.vocab[w]
        lex.is_stop = True

tokenizer = Tokenizer(nlp.vocab)

print('loading glove')
f = open('glove.6B.300d.txt', encoding='utf-8')
model = {}
for line in f:
    splitLine = line.split()
    word = splitLine[0]
    coef = np.asarray(splitLine[1:], dtype='float32')
    model[word] = coef
f.close()
print('done loading glove')


def encode_median_year(event):
    """
    Encode the median of the years found in ci (if years were found).
    We encode the thousands, hundreds and tens
    :param event:
    :return:
    """
    event_years = np.array([])
    tagged = nlp(event)
    ne = list(tagged.ents)
    dates = [entity.text for entity in ne if entity.label_ == 'DATE']
    current_year = datetime.datetime.now().year

    for date in dates:
        date_tagged = nlp(date)
        for word in date_tagged:
            if word.pos_ == 'NUM':
                try:
                    year = parser.parse(word.text).year
                    if year < current_year:
                        event_years = np.append(event_years, year)
                    elif year == current_year and str(current_year) in word.text:
                        # Needed due to problems with small numbers that are not years
                        event_years = np.append(event_years, year)
                except Exception as e:
                    continue
    median = 0
    if event_years.size:
        median = np.median(event_years)
    year_encoding = np.zeros(YEAR_ENCODING_DIM)
    if median:
        # Encode thousands, hundreds and tens of the median of years found in CI
        t = int(median // 1000)
        if 1 <= t <= 2:
            year_encoding[t-1] = 1
            median = median - (1000*t)
            h = int(median // 100)
            if 0 <= h <= 9:
                year_encoding[h + 2] = 1
                median = median - (100 * h)
                d = int(median // 10)
                if 0 <= d <= 9:
                    year_encoding[d + 12] = 1
    return year_encoding


def text_vectorizer(text, embedding_model):
    """
    Get a vector representation of a text.
    The representation is achieved by averaging the embddings of all words in the text.
    """
    sent_vec = np.zeros(shape=(EMB_DIM,))
    existing_words = 0
    for word in text:
        if word.text in embedding_model:
            sent_vec = sent_vec + embedding_model.get(word.text)
            existing_words += 1
    if existing_words > 0:
        return np.asarray(sent_vec) / existing_words
    else:
        return np.asarray(sent_vec)


def create_data_embedding(events, add_year_feat=True, num_concat=6):
    """
    Get the BOE model feature vector.
    :param events: List of events' text
    :param add_year_feat: Whether to add the median year encoding or not
    :param num_concat: We allow concatenating the median year encoding several times.
    :return:
    """
    feat_vecs = []
    tokenized_events = list(tokenizer.pipe(events))
    for idx, event in enumerate(tokenized_events):
        boe_vec = text_vectorizer(event, model)
        if add_year_feat:
            year_feat = encode_median_year(events[idx])
            for i in range(num_concat):
                boe_vec = np.append(boe_vec, year_feat)
        feat_vecs.append(boe_vec)
    return feat_vecs

