from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np
from pprint import pprint
import logging
import re

logger = logging.getLogger(__name__)

class QuoraQPExtractor:
    def __init__(self, file_path, embedding_path=None):
        if file_path is not None:
            with open(file_path) as f:
                self.df = pd.read_csv(f, sep='\t')
                self.q1 = list(self.df.iloc[:, 3])
                self.q2 = list(self.df.iloc[:, 4])
                self.isDuplicate = list(self.df.iloc[:, 5])
        else:
            raise NotImplementedError()

        self.wor = {}
        self.index2word = {}

        self.corpus = self.q1 + self.q2
        self.word_counter = Counter()


        logger.info("Starting Vocab Build")
        self.build_vocab()
        logger.info("Vocab Build Complete")


    def preprocess(self, sentence):
        # print(sentence)
        try:
            return re.sub("[^a-zA-Z0-9]"," ", sentence.lower())
        except:
            print(sentence, " HAS AN ERROR")

    def build_vocab(self):
        logger.info("Building vocab")

        # print(self.corpus)

        for sentence in self.corpus:
            sentence = self.preprocess(sentence)
            self.word_counter.update(sentence.split())

        for i, word in enumerate(self.word_counter.keys()):
            self.wor[word] = i
            self.index2word[i] = word

        self.vocab_size = len(self.wor)
        logger.info("word vocab build complete")

    def get_preprocessed_corpus(self):
        preprocessed_corpus = []
        for sent in self.corpus:
            preprocessed_corpus.append(self.preprocess(sent))

        return preprocessed_corpus


    def get_data(self):

        question_pairs = []
        labels = []
        
        for Question1, Question2, label in zip(self.q1, self.q2, self.isDuplicate):

            question_pairs.append([self.preprocess(Question1), self.preprocess(Question2)])
            # print(Question1)
            # print(label, type(label))
            labels.append(int(label))
        
        return question_pairs, labels

class WikiQAExtractor:
    def __init__(self, file_path, embedding_path=None):
        if file_path is not None:
            with open(file_path) as f:
                self.df = pd.read_csv(f, sep='\t')
                self.queries = list(self.df.iloc[:, 0])
                self.documents = list(self.df.iloc[:, 1])
                self.relations = list(self.df.iloc[:, 2])
        else:
            raise NotImplementedError()

        # TODO add 300k vector for all permutes
        # TODO add option for using word embeddings

        self.word2int, self.int2word = {}, {}
        self.tri2index, self.index2tri = {}, {}
        
        self.word_counter = Counter()
        self.triletter_counter = Counter()

        self.corpus = self.queries +  self.documents
        self.triletter_corpus = []

        logger.info("Starting Vocab Build")
        self.build_vocab()
        logger.info("Vocab Build Complete")

    def preprocess(self, sentence):
        return re.sub("[^a-zA-Z0-9]"," ", sentence.lower())

    def build_vocab(self):
        logger.info("Building vocab")
        for sentence in self.corpus:
            sentence = self.preprocess(sentence)
            self.word_counter.update(sentence.split())

            # update triletter scanning
            tri_sentence = []
            for word in sentence.split():
                tri_word = []
                word = '#' + word + '#'

                for offset in range(0, len(word))[:-2]:
                    triletter = word[offset: offset + 3]
                    tri_word.append(triletter)
                    self.triletter_counter.update([triletter])

                tri_sentence.append(tri_word)

            self.triletter_corpus.append(tri_sentence)

        for i, word in enumerate(self.word_counter.keys()):
            self.word2int[word] = i
            self.int2word[i] = word

        for i, triletter in enumerate(self.triletter_counter.keys()):
            self.tri2index[triletter] = i
            self.index2tri[i] = triletter       

        self.vocab_size = len(self.tri2index)
        logger.info("word vocab build complete")

    def sent2triletter_indexed_sent(self, sentence):
        """Converts a sentence to a triletter sentence
        
        Parameters
        ==========
        sentence:
            A list of words
        """
        triletter_sentence = []
        for word in sentence:
            tri_word = []
            word = '#' + word + '#'

            for offset in range(len(word))[:-2]:
                try:
                    tri_word.append(self.tri2index[word[offset: offset + 3]])
                except :
                    pass
                    # TODO will this clobber some other exceptions ???
                    # maybe an if is in dict would be better but could lead to more
                    # branch misses
                    # logger.info("Found a tri not in dict: %s. Adding it now" % word[offset: offset + 3])

                    # self.tri2index[word[offset: offset + 3]] = self.vocab_size
                    # self.index2tri[self.vocab_size] = word[offset: offset + 3]

                    # self.vocab_size += 1

                    # tri_word.append(self.tri2index[word[offset: offset + 3]])

            triletter_sentence.append(tri_word)

        return triletter_sentence

    def get_term_vector(self, sentence):
        """Converts a sentence into its term vector to be pushed into the neural network
        
        Parameters
        ==========
        sentence:
            A list of words

        Example:
        ========
        >> get_term_vector("how are glaciers formed ?".split())
        [1, 0, 23, 53, ..., 12]
        """
        # TODO check if vocab has been built

        indexed_triletter_sentence = self.sent2triletter_indexed_sent(sentence)

        # TODO vocab size may change due to unseen tris
        # bad coding practice?
        # this line has to be after the above line
        vector = np.zeros(self.vocab_size)


        for word in indexed_triletter_sentence:
            for triletter in word:
                vector[triletter] += 1
        return vector.reshape(self.vocab_size,)


    def make_indexed_corpus(self):
        logger.info('making indexed triletter corpus')

        self.indexed_triletter_corpus = []

        for sentence in self.triletter_corpus:
            indexed_tri_sentence = []
            for word in sentence:
                indexed_tri_word = []
                for triletter in word:
                    int_tri_word.append(self.tri2indexed[triletter])
                indexed_tri_sentence.append(int_tri_word)
            indexed_triletter_corpus.append(int_tri_sentence)

        logger.info('indexed triletter corpus made')

        return indexed_triletter_corpus

    def get_X_y(self, batch_size=32):

        # TODO Implement batch sizing

        queries = []
        docs = []
        labels = []

        for Question, Answer in self.df.groupby('QuestionID').apply(dict).items():

            query_group = []
            label_group = []

            for q, d, l in zip(Answer['Question'], Answer['Sentence'], Answer['Label']):
                queries.append(self.get_term_vector(self.preprocess(q)))
                docs.append(self.get_term_vector(self.preprocess(d)))
                labels.append(l)
        
        queries = np.array(queries)
        docs = np.array(docs)
        labels = np.array(labels)

        return queries, docs, labels

    def get_data(self):

        self.questions = []

        for Question, Answer in self.df.groupby('QuestionID').apply(dict).items():

            document_group = []

            query = []
            doc = []
            label = []

            for q, d, l in zip(Answer['Question'], Answer['Sentence'], Answer['Label']):
                document_group.append([self.preprocess(q), self.preprocess(d), l])

            self.questions.append(document_group)
        
        return self.questions