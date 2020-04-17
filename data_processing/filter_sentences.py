import spacy
import get_wikipedia_data

nlp = spacy.load('en_core_web_lg')


def filer_sent(article_sentences, filter_func):
    return list(filter(filter_func, article_sentences))

# Next 3 function construct a filter that extracts
# from a given article sentences containing a date.

def has_date(sentence):
    nes = nlp(sentence).ents
    for ent in nes:
        if ent.label_ == 'DATE':
            return True
    return False


def has_date_filter():
    return lambda sentence: has_date(sentence)


def get_sentences_with_date(articles_sentences):
    filtered = [None] * len(articles_sentences)
    for i in range(len(articles_sentences)):
        filtered[i] = (filer_sent(articles_sentences[i], has_date_filter()))
    return filtered


# Next 3 function construct a filter that extracts from a given article sentences that
# contain at least one extracted title, that is not the title of the article itself

def has_one_extracted_word(sentence, all_key_ents, current_ent):
    for key_ent in all_key_ents:
        if key_ent in sentence:
            if key_ent.find(current_ent) == -1 and current_ent.find(key_ent) == -1:
                nes1 = nlp(key_ent).ents
                nes2 = nlp(current_ent).ents
                if len(nes1) != 0 and len(nes2) != 0:
                    # In the specific case were both key entities are geopolitical entities,
                    # We drop those sentences. This is due to the fact the sentences extracted
                    # using two geopolitical entities tend to be too general and
                    # are not entity-specific
                    if not (nes1[0].label_ == 'GPE' and nes2[0].label_ == 'GPE'):
                        return True
                else:
                    return True
    return False


def has_one_extracted_word_filter(other_key_ents, current_ent):
    return lambda sentence: has_one_extracted_word(sentence, other_key_ents, current_ent)


def get_sentences_with_other(articles_sentences, key_ents):
    filtered = [None] * len(articles_sentences)
    for i in range(len(articles_sentences)):
        all_but_curr_ent = key_ents[:i] + key_ents[i + 1:]
        filtered[i] = (filer_sent(articles_sentences[i],
                                  has_one_extracted_word_filter(all_but_curr_ent, key_ents[i])))
    return filtered


# Next 3 function construct a filter that extracts from a given article sentences that
# contain at least one extracted title, that is not the title of the article
# itself, as well as one of the extracted actions.
def has_one_other_and_action(sentence, other_key_ents, actions):
    tagged = nlp(sentence)
    lemmatized = [tok.lemma_ for tok in tagged]
    found_other_ent = False
    if len(other_key_ents) == 0:
        # The original key entities list only had one entity for the event -
        # just search for sentences with one of the actions
        found_other_ent = True
    for ent in other_key_ents:
        if ent in sentence:
            # Sentence has at least one other entity
            found_other_ent = True
    if not found_other_ent:
        return False
    for action in actions:
        if action in lemmatized:
            return True
    return False


def has_one_other_and_action_filter(other_key_ents, actions):
    return lambda sentence: has_one_other_and_action(sentence, other_key_ents, actions)


def get_sentences_with_other_and_action(articles_sentences, key_ents, actions):
    filtered = [None] * len(articles_sentences)
    for i in range(len(articles_sentences)):
        all_but_curr_ent = key_ents[:i] + key_ents[i + 1:]
        filtered[i] = (filer_sent(articles_sentences[i],
                                  has_one_other_and_action_filter(all_but_curr_ent, actions)))
    return filtered


# Next 3 function construct a filter that extracts from a given article sentences
# that contain all extracted titles, except maybe the title of the article itself
def has_all_extracted_words(sentence, other_key_ents):
    if len(other_key_ents) == 0:
        return False
    for ent in other_key_ents:
        if ent not in sentence:
            return False
    return True


def has_all_extracted_words_filter(other_key_ents):
    return lambda sentence: has_all_extracted_words(sentence, other_key_ents)


def get_sentences_with_all_others(articles_sentences, key_ents):
    filtered = [None] * len(articles_sentences)
    for i in range(len(articles_sentences)):
        all_but_curr_ent = key_ents[:i] + key_ents[i + 1:]
        filtered[i] = filer_sent(articles_sentences[i],
                                 has_all_extracted_words_filter(all_but_curr_ent))
    return filtered


# Next 3 function construct a filter that extracts from a given article sentences that
# contain at least one extracted title, that is not the title of the article
# itself, as well as the title of the article itself
def has_title_and_other(sentence, current_ent, other_key_ents):
    for other_ent in other_key_ents:
        if other_ent in sentence and current_ent in sentence:
            return True
    return False


def has_title_and_other_filter(current_ent, other_key_ents):
    return lambda sentence: has_title_and_other(sentence, current_ent, other_key_ents)


def get_sentences_with_ent_and_other(articles_sentences, key_ents):
    filtered = [None] * len(articles_sentences)
    for i in range(len(articles_sentences)):
        all_but_curr_ent = key_ents[:i] + key_ents[i + 1:]
        filtered[i] = filer_sent(articles_sentences[i], has_title_and_other_filter(key_ents[i],
                                                                                   all_but_curr_ent))
    return filtered