import spacy
from spacy.tokenizer import Tokenizer
import re
import dateutil.parser as parser
import datetime
import numpy as np
from boe_classifier import train, test
from preprocess_pickle import get_text_and_labels, ablation_A_get_text_and_labels


nlp = spacy.load('en_core_web_lg')

# stop words missing for en_core_web_lg for spaCy v2.0, see https://github.com/explosion/spaCy/issues/1574
# A workaround:
for word in nlp.Defaults.stop_words:
    for w in (word, word[0].upper() + word[1:], word.upper()):
        lex = nlp.vocab[w]
        lex.is_stop = True

tokenizer = Tokenizer(nlp.vocab)


def find_years(text):
    """
    Returns a list of all the years in a text.
    Years are kept as strings in the returned list.
    """
    event_years = []
    tagged = nlp(text)
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
                        event_years.append(str(year))
                    elif year == current_year and str(current_year) in word.text:
                        # Needed due to problems with small numbers that are not years
                        event_years.append(str(year))
                except Exception as e:
                    continue
    return event_years


def remove_numbers(events):
    cleaned = []
    for event in events:
        events_tok = nlp(event)
        no_numbers = ' '.join(token.lemma_ for token in events_tok if not token.like_num)
        no_numbers = re.sub(' +', ' ', no_numbers)
        cleaned.append(no_numbers)
    return cleaned


def remove_years(events):
    cleaned = []
    for event in events:
        events_tok = nlp(event)
        event_years = find_years(event)
        no_years = ' '.join(token.lemma_ for token in events_tok
                                if token.text not in event_years)
        no_years = re.sub(' +', ' ', no_years)
        cleaned.append(no_years)
    return cleaned


def remove_dates(events):
    cleaned = []
    for event in events:
        events_tok = nlp(event)
        no_date = ' '.join(token.lemma_ for token in events_tok
                                if token.ent_type_ != 'DATE')
        no_date = re.sub(' +', ' ', no_date)
        cleaned.append(no_date)
    return cleaned


if __name__ == "__main__":
    wotd_train_with_ci, wotd_train_no_ci, wotd_train_labels = get_text_and_labels('data/train/wotd.pkl')
    wotd_dev_with_ci, wotd_dev_no_ci, wotd_dev_labels = get_text_and_labels('data/validation/wotd.pkl')

    ci_extracted_only_train, ci_extracted_only_no_description_train, \
            ci_extracted_only_train_labels = ablation_A_get_text_and_labels('data/train/wotd.pkl')
    ci_extracted_only_dev, ci_extracted_only_no_description_dev, \
            ci_extracted_only_dev_labels = ablation_A_get_text_and_labels('data/validation/wotd.pkl')

    # Ablation A - only events for which CI was extracted are used
    print("results for dev - ablation A")
    clf_trained = train(ci_extracted_only_train, ci_extracted_only_train_labels)
    test(ci_extracted_only_dev, ci_extracted_only_dev_labels, clf_trained)

    print("-event description")
    clf_trained = train(ci_extracted_only_no_description_train, ci_extracted_only_train_labels)
    test(ci_extracted_only_no_description_dev, ci_extracted_only_dev_labels, clf_trained)

    # Ablation B - all events included
    print("results for dev - full data")
    clf_trained = train(wotd_train_with_ci, wotd_train_labels)
    test(wotd_dev_with_ci, wotd_dev_labels, clf_trained)

    print("no numbers")
    clf_trained = train(remove_numbers(wotd_train_with_ci), wotd_train_labels)
    test(remove_numbers(wotd_dev_with_ci), wotd_dev_labels, clf_trained)

    print("no years")
    clf_trained = train(remove_years(wotd_train_with_ci), wotd_train_labels)
    test(remove_years(wotd_dev_with_ci), wotd_dev_labels, clf_trained)

    print("no dates")
    clf_trained = train(remove_dates(wotd_train_with_ci), wotd_train_labels)
    test(remove_dates(wotd_dev_with_ci), wotd_dev_labels, clf_trained)
