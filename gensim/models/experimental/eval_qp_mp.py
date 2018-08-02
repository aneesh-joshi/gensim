from gensim import downloader as api
from sklearn.utils import shuffle
from matchpyramid import MatchPyramid
import re

train_split = 0.8

qqp = api.load('quora-duplicate-questions')

def preprocess(sent):
	return re.sub("[^a-zA-Z0-9]", " ", sent.strip().lower()).split()

q1, q2, duplicate = [], [], []
for row in qqp:
	q1.append(preprocess(row['question1']))
	q2.append(preprocess(row['question2']))
	duplicate.append(int(row['is_duplicate']))

print('Number of question pairs', len(q1))
print('Number of duplicates', sum(duplicate))
print('% duplicates', 100.*sum(duplicate)/len(q1))
print('-----------------------------------------')

q1, q2, duplicate = shuffle(q1, q2, duplicate)

train_q1, test_q1 = q1[:int(len(q1)*train_split)], q1[int(len(q1)*train_split):]
train_q2, test_q2 = q2[:int(len(q2)*train_split)], q2[int(len(q2)*train_split):]
train_duplicate, test_duplicate = duplicate[:int(len(duplicate)*train_split)], duplicate[int(len(duplicate)*train_split):]

assert len(train_q1) == len(train_duplicate)
assert len(test_q2) == len(test_duplicate)


print('Number of question pairs in train', len(train_q1))
print('Number of duplicates in train', sum(train_duplicate))
print('%% duplicates', 100.*sum(train_duplicate)/len(train_q1))
print('-----------------------------------------')

print('Number of question pairs in test', len(test_q1))
print('Number of duplicates in test', sum(test_duplicate))
print('%% duplicates', 100.*sum(test_duplicate)/len(test_q1))
print('-----------------------------------------')

print(test_q1)

kv_model = api.load('glove-wiki-gigaword-50')

mp_model = MatchPyramid(queries=q1, docs=q2, labels=duplicate, target_mode='classification', word_embedding=kv_model)