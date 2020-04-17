import spacy
from spacy.tokenizer import Tokenizer
import re

nlp = spacy.load('en_core_web_lg')

# stop words missing for en_core_web_lg for spaCy v2.0, see https://github.com/explosion/spaCy/issues/1574
# A workaround:
for word in nlp.Defaults.stop_words:
    for w in (word, word[0].upper() + word[1:], word.upper()):
        lex = nlp.vocab[w]
        lex.is_stop = True

tokenizer = Tokenizer(nlp.vocab)


def clean_str(str_to_clean):
    cleaned = re.sub('\n', ' ', str_to_clean)
    cleaned = ''.join(c for c in cleaned)
    cleaned = re.sub('[=/\[\]\-]', ' ', cleaned)
    cleaned = re.sub(' +', ' ', cleaned)
    return cleaned


def linguistic_clean(event):
    events_tok = nlp(event)
    cleaned = ' '.join(token.lemma_.lower() for token in events_tok
                            if not (token.is_stop or token.is_punct))
    cleaned = re.sub(' +', ' ', cleaned)
    return cleaned