import sys
import os
from pprint import pprint
import csv
import re
from matchpyramid import MatchPyramid
import gensim.downloader as api
import argparse

import csv
import sys
sys.path.append('..')
from gensim.utils import simple_preprocess

Q1_INDEX = 3
Q2_INDEX = 4
LABEL_INDEX = 5

with open('quora_duplicate_questions.tsv', encoding='utf8') as tsv_file:
    tsv_reader = csv.reader(tsv_file, delimiter='\t', quoting=csv.QUOTE_NONE)
    Q1, Q2, labels = [], [], []
    first = True
    for row in tsv_reader:
        if first:
            first = False
            continue
        Q1.append(simple_preprocess(row[Q1_INDEX]))
        Q2.append(simple_preprocess(row[Q2_INDEX]))
        labels.append(row[LABEL_INDEX])

# for q1, q2, l in zip(Q1, Q2, labels):
#     print(q1, q2, l)

num_samples = len(q1)
train_x1, train_x2, train_l = Q1[:0.8*num_samples], Q2[:0.8*num_samples], labels[:0.8*num_samples]
test_x1, test_x2, test_l = Q1[0.8*num_samples:], Q2[0.8*num_samples:], labels[0.8*num_samples:]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', help='number of epochs to train')
    parser.add_argument('--w2v_dim')
    parser.add_argument('--text_maxlen')
    parser.add_argument('--model_save_name')

    args = parser.parse_args()

    epochs = int(args.epochs)
    w2v_dim = args.w2v_dim
    text_maxlen = int(args.text_maxlen)
    model_save_name = args.model_save_name

    kv_model = api.load('glove-wiki-gigaword-' + str(w2v_dim))

    # Train the model
    mp_model = MatchPyramid(
                    queries=q_iterable, docs=d_iterable, labels=l_iterable, word_embedding=kv_model, epochs=epochs, 
                    text_maxlen=text_maxlen #validation_data=[q_val_iterable, d_val_iterable, l_val_iterable],
                )

    print('Test set results')
    mp_model.evaluate(q_test_iterable, d_test_iterable, l_test_iterable)

    if model_save_name is not None:
        mp_model.save(model_save_name)