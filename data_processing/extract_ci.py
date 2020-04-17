import itertools
import argparse
import sys
import re
import pandas as pd
import filter_sentences, get_wikipedia_data


def extract_filtered_sentences(event):
    articles_sents, key_ents, actions = get_wikipedia_data.get_event_ci(event)

    # First, try the filter that searches for sentences with another key entity and an action
    extracted_ci = filter_sentences.get_sentences_with_other_and_action(articles_sents, key_ents, actions)
    extracted_ci = [sent for sent in extracted_ci if len(sent) != 0]

    if len(extracted_ci) == 0:
        # If no sentences were found using the first filter,
        # search for sentences with another key entity
        extracted_ci = filter_sentences.get_sentences_with_other(articles_sents, key_ents)
        extracted_ci = [sent for sent in extracted_ci if len(sent) != 0]
    extracted_ci = list(itertools.chain.from_iterable(extracted_ci))

    extracted_ci_str = ' '.join(extracted_ci)
    extracted_ci_str = re.sub('\n', ' ', extracted_ci_str)

    return extracted_ci_str


def save_ci(input_file, output_file):
    ci = []
    df = pd.read_pickle(input_file)
    events = df['Event'].values

    for event in events:
        ci.append(extract_filtered_sentences(event))

    df['Information'] = ci
    df.to_pickle(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get an input pickle files that contains historical '
                                                 'events, and extract ci for the events')
    parser.add_argument("infile", help="path to input pickle file")
    parser.add_argument("outfile", help="name for output file")
    args = parser.parse_args()
    save_ci(sys.argv[1], sys.argv[2])
