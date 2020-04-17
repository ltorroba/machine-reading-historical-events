import spacy
import wikipedia
import re

nlp = spacy.load('en_core_web_lg')

NE_UNWANTED_LABLES = ['MONEY', 'TIME', 'PERCENT', 'DATE', 'ORDINAL', 'QUANTITY', 'CARDINAL', 'NORP']
REL_PRONOUNS = ['which', 'that', 'whom', 'whose', 'who']
UNWANTED_CATEGORIES = 'Given names'
SECTIONS_TO_REMOVE = ['See also', 'References', 'Bibliography', 'External links']
AUX_VERBS = ['be', 'begin', 'become']


def get_full_ent(tagged_event, title_root):
    tokens = tagged_event[title_root.left_edge.i: title_root.right_edge.i + 1]
    for i in range(len(tokens)):
        # Trim the title if encountered a verb or in case of relative clause. For example:
        # 'Alaska, which is the..' - the title should only contain Alaska
        if tokens[i].text in REL_PRONOUNS or tokens[i].pos_ == 'VERB':
            tokens = tokens[:i]
    filtered = [el.text for el in tokens if el.pos_ != 'ADV' and el.pos_ != 'PUNCT'
                   and el.pos_ != 'DET']
    return ' '.join(filtered)


def part_of_ent(other, ents):
    for ent in ents:
        if other in ent:
            return True
    return False


def get_named_ents(tagged_event):
    ne = list(tagged_event.ents)
    named_ents = [entity.text for entity in ne if entity.label_ not in NE_UNWANTED_LABLES]
    for i in range(len(named_ents)):
        named_ents[i] = named_ents[i].replace("the ", "")
        named_ents[i] = named_ents[i].replace("The ", "")
    return named_ents


def get_subj_obj_ents(word, event_tagged):
    if (word.dep_ == 'nsubj' and word.pos_ != 'PRON' and word.pos_ != 'ADJ') \
            or (word.dep_ == 'nsubjpass' and word.pos_ == 'PROPN'):
        # Adds the subject of the sentence
        subj = get_full_ent(event_tagged, word)
        return subj

    if word.dep_ == 'dobj' and word.left_edge.dep_ != 'poss' and word.pos_ != 'NOUN' \
            and word.pos_ != 'PRON':
        # Adds the object of the sentence
        obj = get_full_ent(event_tagged, word)
        return obj


def get_entities_and_actions(event):
    """
    Gets an event description and extracts key entities and actions.
    """
    actions = []
    tagged_event = nlp(event)

    # First, all named entities are marked as titles
    ents = get_named_ents(tagged_event)

    for word in tagged_event:
        # Actions
        if word.pos_ == 'VERB' and word.lemma_ not in AUX_VERBS:
            actions.append(word.lemma_)

        # Entities
        if word.dep_ == 'appos' and word.head.text in ents:
            # In cases such as: Columbus, New Mexico, both Columbus and New Mexico are named
            # entities, but we would like to extract as title 'Columbus New Mexico'
            full = get_full_ent(tagged_event, word)
            # Check if parts of the wanted title was already included in our list of titles.
            # If so, remove those parts, so only the final full title will be included
            if full in ents:
                ents.remove(full)
            ents.remove(word.head.text)
            ents.append(get_full_ent(tagged_event, word.head))

        if (word.dep_ == 'compound' and (word.head.dep_ == 'ROOT' or
                                         word.head.head.dep_ == 'ROOT')) \
                or ((word.dep_ == 'amod' or word.dep_ == 'nmod') and word.head.dep_ == 'ROOT'
                    and word.head.pos_ != 'VERB'):
            # First part of the condition: get compounds whose head is the root.
            # Second part: In case that the root is not a verb. For example -
            # Yugoslav Wars: Srebrenica massacre begins..
            if word.left_edge.i < word.head.i:
                title = tagged_event[word.left_edge.i: word.head.i + 1].text
            else:
                title = tagged_event[word.head.i: word.right_edge.i + 1].text
            if not part_of_ent(title, ents):
                ents.append(title)

        subj_or_obj_ent = get_subj_obj_ents(word, tagged_event)
        if subj_or_obj_ent and subj_or_obj_ent not in ents:
            ents.append(subj_or_obj_ent)

    return ents, actions


def get_event_articles(ents):
    """
    Gets a list of key entities and extracts relevant Wikipedia articles.
    """
    articles = []
    # Wikipedia articles titles might be slightly different form the extracted titles.
    # In such cases - keep the title as at appears in Wikipedia
    articles_titles = []

    for ent in ents:
        try:
            article = wikipedia.page(ent)
            if article not in articles and UNWANTED_CATEGORIES not in article.categories:
                articles.append(article)
                exact_title = re.sub(r' *\([ a-zA-Z_0-9]*\)', '', article.title)
                articles_titles.append(exact_title)
        except wikipedia.DisambiguationError as e:
            # More than one option for an article with this title -
            # takes the first Wikipedia suggestion
            ent = e.options[0]
            try:
                article = wikipedia.page(ent)
                if article not in articles and UNWANTED_CATEGORIES not in article.categories:
                    articles.append(article)
                    exact_title = re.sub(r' *\([ a-zA-Z_0-9]*\)', '', article.title)
                    articles_titles.append(exact_title)
            except Exception as e:
                print(e)
        except Exception as e:
            print(e)
    return articles, articles_titles


def get_event_ci(event):  # TODO was get_article_sentences
    """
    Gets an event and returns:
    i. A list of lists. Each sub-list is a list of sentences extracted from a Wikipedia article
    ii. The extracted key entities for the event
    iii. The extracted actions of the event
    :param event: A string - the event description
    """
    primary_titles, actions = get_entities_and_actions(event)
    articles, titles = get_event_articles(primary_titles)
    all_sentences = [None] * len(articles)
    for i, article in enumerate(articles):
        sections = article.sections
        content = article.content

        for sec in sections:
            if sec in SECTIONS_TO_REMOVE:
                section = article.section(sec)
                if section is not None:
                    content = content.replace(section, '')

        tagged_content = nlp(content)
        article_sentences = list(tagged_content.sents)
        # Remove Wikipedia headlines (starts with '==')
        article_sentences = [sent.text for sent in article_sentences if '==' not in sent.text]
        all_sentences[i] = article_sentences

    return all_sentences, titles, actions