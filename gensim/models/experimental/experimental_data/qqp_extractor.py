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

for q1, q2, l in zip(Q1, Q2, labels):
	print(q1, q2, l)