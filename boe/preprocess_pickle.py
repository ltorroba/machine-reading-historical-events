import process_text
import pandas as pd
import numpy as np


def get_text_and_labels(file_name):
    data = pd.read_pickle(file_name)
    ci = data['Information'].values
    descriptions = data['Event'].values
    labels = data['YY'].values

    join_processed = []
    descriptions_processed = []
    for i, description in enumerate(descriptions):
        if i % 500 == 0:
            print("Processed {} events".format(i))
        s = description
        if len(ci[i]):
            extracted = ' '.join(ci[i])
            s = description + ' ' + extracted
        join_processed.append(process_text.linguistic_clean(process_text.clean_str(s)))
        descriptions_processed.append(process_text.linguistic_clean(process_text.clean_str(description)))

    return join_processed, descriptions_processed, labels


def ablation_A_get_text_and_labels(file_name):
    """
    Keep only events and events labels for cases in which CI was extracted
    :param file_name:
    :return:
    """
    data = pd.read_pickle(file_name)
    ci = data['Information'].values
    descriptions = data['Event'].values
    labels = data['YY'].values

    join_processed = []
    ci_only = []
    filtered_labels = []
    for i, description in enumerate(descriptions):
        if i % 500 == 0:
            print("Processed {} events".format(i))
        if len(ci[i]):
            extracted = ' '.join(ci[i])
            s = description + ' ' + extracted
            join_processed.append(process_text.linguistic_clean(process_text.clean_str(s)))
            ci_only.append(process_text.linguistic_clean(process_text.clean_str(extracted)))
            filtered_labels.append(labels[i])

    return join_processed, ci_only, np.array(filtered_labels)