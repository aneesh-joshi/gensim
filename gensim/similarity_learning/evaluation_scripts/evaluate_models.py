from gensim.similarity_learning import WikiQAExtractor
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
import numpy as np
import pandas as pd
import argparse
import os
import logging
import time

logger = logging.getLogger(__name__)

"""
This script should be run to get a model by model based or full evaluation
Make sure you run gensim/similarity_learning/data/get_data.py to get the datasets

Example usage:
==============

For evaluating doc2vec on the WikiQA corpus
$ python evaluate_models.py --model doc2vec --datapath ../data/WikiQACorpus/

For evaluating word2vec averaging on the WikiQA corpus
$ python evaluate_models.py --model word2vec --datapath ../data/WikiQACorpus/ --word_embedding_path ../evaluation_scripts/glove.6B.50d.txt  # noqa:F401

For evaluating the TREC format file produced by MatchZoo:
$ python evaluate_models.py  --model mz --mz_result_file predict.test.wikiqa.txtDRMM
Note: here "predict.test.wikiqa.txtDRMM" is the file output by MZ. It has been provided in this repo as an example.

For evaluating all models
-with one mz output file
$ python evaluate_models.py --model all --mz_result_file predict.test.wikiqa.txtDRMM --result_save_path results --word_embedding_path ../evaluation_scripts/glove.6B.50d.txt --datapath ../data/WikiQACorpus/  # noqa:F401
-with a mz folder filled with result files
$ python evaluate_models.py  --model all --mz_result_folder mz_results/ --result_save_path results_all --datapath ../data/WikiQACorpus/ --word_embedding_path ../evaluation_scripts/glove.6B.50d.txt  # noqa:F401
"""


class LabeledLineSentence(object):
    """class to make sentences iterable
    """

    def __init__(self, corpus):
        self.corpus = corpus

    def __iter__(self):
        for uid, line in enumerate(self.corpus):
            yield TaggedDocument(words=line.split(), tags=['SENT_%s' % uid])


# list to store results from all models to be saved later
results_list = []


def mapk(Y_true, Y_pred):
    """Function to get Mean Average Precision(MAP) for a given set of Y_true, Y_pred
    TODO Currently doesn't support mapping at k. Couldn't use only map as it's a
    reserved word

    parameters:
    ===========
    Y_true : numpy array of ints either 1 or 0
        Contains the true, ground truth values of the queries
        Example: [[0, 1, 0, 1],
                  [0, 0, 0, 0, 1, 0],
                  [0, 1, 0]
                 ]

    Y_pred : numpy array of floats between -1 and 1
        Contains the predicted cosine similarity values of the queries
        Example: [
                  [0.1, , -0.01, 0.4],
                  [0.12, -0.43, 0.2, 0.1, 0.99, 0.7],
                  [0.5, 0.63, 0.92]
                 ]
    """
    aps = []
    n_skipped = 0
    for y_true, y_pred in zip(Y_true, Y_pred):
        # skip datapoints where there is no solution
        if np.sum(y_true) < 1:
            n_skipped += 1
            continue

        pred_sorted = sorted(zip(y_true, y_pred),
                             key=lambda x: x[1], reverse=True)
        avg = 0
        n_relevant = 0

        for i, val in enumerate(pred_sorted):
            if val[0] == 1:
                avg += 1. / (i + 1.)
                n_relevant += 1

        if n_relevant != 0:
            ap = avg / n_relevant
            aps.append(ap)

    logger.info("Skipped %d out of %d data points" % (n_skipped, len(Y_true)))
    return np.mean(np.array(aps))


def mean_ndcg(Y_true, Y_pred, k=10):
    """Calculates the mean discounted normalized cumulative gain over all
    the entries limited to the integer k

    parameters:
    ===========
    Y_true : numpy array of floats giving the rank of document for a given query
        Contains the true, ground truth values of the queries
        Example: [
                  [0, 1, 0, 1],
                  [0, 0, 0, 0, 1, 0],
                  [0, 1, 0]
                 ]

    Y_pred : numpy array of floats between -1 and 1
        Contains the predicted cosine similarity values of the queries
        Example: [[0.1, , -0.01, 0.4],
                  [0.12, -0.43, 0.2, 0.1, 0.99, 0.7],
                  [0.5, 0.63, 0.92]
                 ]
    """
    ndcgs = []
    n_skipped = 0
    for y_true, y_pred in zip(Y_true, Y_pred):
        if np.sum(y_true) < 1:
            n_skipped += 1
            continue

        pred_sorted = sorted(zip(y_true, y_pred),
                             key=lambda x: x[1], reverse=True)
        true_sorted = sorted(zip(y_true, y_pred),
                             key=lambda x: x[0], reverse=True)

        pred_sorted = pred_sorted[:k]
        true_sorted = true_sorted[:k]

        dcg = 0
        for i, val in enumerate(pred_sorted):
            if val[0] == 1:
                dcg += 1. / np.log2(i + 2)

        idcg = 0
        for i, val in enumerate(true_sorted):
            if val[0] == 1:
                idcg += 1. / np.log2(i + 2)

        if idcg != 0:
            ndcgs.append(dcg / idcg)
    logger.info("Skipped %d out of %d data points" % (n_skipped, len(Y_true)))
    return np.mean(np.array(ndcgs))


def accuracy(Y_true, Y_pred):
    """Calculates accuracy as (number of correct predictions / number of predictions)
    WARNING: TODO this definition of accuracy doesn't allow for two correct answers
    """
    n_correct = 0
    for y_pred, y_true in zip(Y_pred, Y_true):
        if (np.argmax(y_true) == np.argmax(y_pred)):
            n_correct += 1
    return n_correct / len(Y_true)


def cos_sim(vec1, vec2):
    """Calculates the cosine similarity of 2 vectos
    """
    return np.sum(vec1 * vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def get_metric_results(Y_true, Y_pred, k_range=[1, 3, 5, 10, 20]):
    """returns a dict of calculated metrics
    """
    eval_metrics = {}
    eval_metrics["map"] = mapk(Y_true, Y_pred)

    for k in k_range:
        eval_metrics["ndcg@" + str(k)] = mean_ndcg(Y_true, Y_pred, k=k)

    return eval_metrics


def doc2vec_eval(datapath, vec_size=20, alpha=0.025):
    """Trains the doc2vec model on training data of WikiQA and then
    evaluates on test data

    parameters:
    ==========

    datapath : string
        path to the WikiQA folder

    vec_size : int
        size of the hidden layer

    alpha : float
        The initial learning rate.
    """
    # load testing data
    wikiqa_train = WikiQAExtractor(os.path.join(datapath, "WikiQA-test.tsv"))
    train_data = wikiqa_train.get_preprocessed_corpus()
    lls = LabeledLineSentence(train_data)

    initial_time = time.time()
    logger.info("Building and training doc2vec model")
    model = Doc2Vec(size=vec_size,
                    alpha=alpha,
                    min_alpha=0.025,
                    min_count=1,
                    dm=1,
                    iter=50)
    model.build_vocab(lls)
    model.train(lls,
                total_examples=model.corpus_count,
                epochs=model.iter)
    logger.info("Building and training of doc2vec done")
    logger.info("Time taken to train is %f" %
                float(time.time() - initial_time))

    # load test data
    wikiqa_test = WikiQAExtractor(os.path.join(datapath, "WikiQA-test.tsv"))
    test_doc_data = wikiqa_test.get_data()
    Y_true = []
    Y_pred = []

    for doc in test_doc_data:
        y_true = []
        y_pred = []
        for query, d, label in doc:
            y_pred.append(cos_sim(model.infer_vector(
                query), model.infer_vector(d)))
            y_true.append(label)
        Y_true.append(y_true)
        Y_pred.append(y_pred)
    Y_pred = np.array(Y_pred)
    Y_true = np.array(Y_true)

    results = get_metric_results(Y_true, Y_pred)
    results["method"] = "d2v"
    print("Results of evaluating on doc2vec:")
    print(results)

    results_list.append(results)

    # TODO might have to tune with validation data
    # TODO get params from user?


def word2vec_eval(datapath, word_embedding_path):
    """Averages words in a query to represent the sentence/doc
    If the word is out of vocabulary, we ignore it

    parameters:
    ==========
    datapath : string
        path to the WikiQA folder

    word_embedding_path : string
        path to the the .txt which has the Glove word embeddings
    """
    # load test data
    # Note: here we are not using train data to keep results consistent with
    # other models
    wikiqa_test = WikiQAExtractor(os.path.join(datapath, "WikiQA-test.tsv"))
    test_doc_data = wikiqa_test.get_data()

    logger.info("Starting building word-vec dict")
    # dict to store word-index pairs
    w2v = {}
    with open(word_embedding_path) as f:
        for line in f:
            string_array = np.array(line.split()[1:])
            string_array = [float(i) for i in string_array]
            w2v[line.split()[0]] = string_array
    logger.info("Word-vec dict build complete")

    def sent2vec(w2v, sentence):
        """Function to convert a sentence into an averaged vector
        """
        vec_sum = []
        for word in sentence.split():
            if word in w2v:
                vec_sum.append(w2v[word])
        return np.mean(np.array(vec_sum), axis=0)

    Y_true = []
    Y_pred = []
    for query_doc_group in test_doc_data:
        y_true = []
        y_pred = []
        for query, doc, label in query_doc_group:
            y_pred.append(cos_sim(sent2vec(w2v, query), sent2vec(w2v, doc)))
            y_true.append(label)
        Y_true.append(y_true)
        Y_pred.append(y_pred)
    Y_pred = np.array(Y_pred)
    Y_true = np.array(Y_true)

    results = get_metric_results(Y_true, Y_pred)
    results["method"] = "w2v"

    print("Results of evaluating on word2vec:")
    print(results)

    results_list.append(results)
    # TODO we can do an evaluation on the whole dataset since this is unsupervised
    # currently WikiQAExtractor cannot support multiple files. Add it.
    # call it get_unsupervised_data or something

    # TODO add option to train on multiple w2v dimension files
    # Example: [50d, 100d, 200d, etc]

    # TODO maybe replace w2v dict which is memory intensice with gensim
    # KeyedVectors


def mz_eval(mz_output_file):
    """Evaluates the metrics on a TREC format file output by MatchZoo

    parameters:
    ==========
    mz_output_file : string
        path to MatchZoo output TREC format file
    """

    with open(mz_output_file) as f:
        df = pd.read_csv(f, sep='\t', names=[
            "QuestionID", "Q0", "Doc ID", "Doc No", "predicted_score", "model name", "actual_score"])

    Y_true = []
    Y_pred = []

    # Group the results based on QuestionID column
    for Question, Answer in df.groupby('QuestionID').apply(dict).items():
        y_true = []
        y_pred = []

        for d, l in zip(Answer['predicted_score'], Answer['actual_score']):
            y_pred.append(d)
            y_true.append(l)

        Y_pred.append(y_pred)
        Y_true.append(y_true)

    results = get_metric_results(Y_true, Y_pred)
    results["method"] = "MZ"  # TODO add a way to specify the function

    print("Results of evaluating on mz_eval:")
    print(results)

    results_list.append(results)


def mz_eval_multiple(mz_output_file_dir):
    """Evaluates multiple TREC format file output by MatchZoo

    parameters:
    ==========
    mz_output_file_dir : string
        path to folder with MatchZoo output TREC format files
    """

    for mz_output_file in os.listdir(mz_output_file_dir):
        with open(os.path.join(mz_output_file_dir, mz_output_file)) as f:
            df = pd.read_csv(f, sep='\t', names=[
                             "QuestionID", "Q0", "Doc ID", "Doc No", "predicted_score", "model name", "actual_score"])

        Y_true = []
        Y_pred = []

        # Group the results based on QuestionID column
        for Question, Answer in df.groupby('QuestionID').apply(dict).items():
            y_true = []
            y_pred = []

            for d, l in zip(Answer['predicted_score'], Answer['actual_score']):
                y_pred.append(d)
                y_true.append(l)

            Y_pred.append(y_pred)
            Y_true.append(y_true)

        results = get_metric_results(Y_true, Y_pred)
        results["method"] = mz_output_file.split('.')[2]

        print("Results of evaluating on mz_eval:")
        print(results)

        results_list.append(results)


def write_results_to_file(results_list, file_to_write, k_range=[1, 3, 5, 10, 20]):
    """Writes the evaluated metrics in the given
    """
    k_range = ["ndcg@" + str(k) for k in k_range]
    metric_order = ["map"] + k_range

    header = "Method, "
    for metric in metric_order:
        header += metric + ", "
    header += "\n"

    to_write = ""
    for result in results_list:
        to_write += result["method"] + ", "
        for metric in metric_order:
            to_write += str(result[metric]) + ", "
        to_write += "\n"

    with open(file_to_write + ".csv", "w") as f:
        f.write(header)
        f.write(to_write)

    print("Results saved in %s" % file_to_write + ".csv")


if __name__ == '__main__':

    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        default='all',
                        help='runs the evaluation of doc2vec')

    # Note: we currently only support WikiQA
    parser.add_argument('--datapath',
                        help='path to the folder with WikiQACorpus. Path should include WikiQACorpus\
                         Make sure you have run get_data.py in gensim/similarity_learning/data/')

    # TODO include gensim-data path to word embeddings
    parser.add_argument('--word_embedding_path',
                        help='path to the Glove word embedding file')

    parser.add_argument('--mz_result_file',
                        help='path to the prediction output file made by mz')

    parser.add_argument('--result_save_path',
                        default=None,
                        help='path to save the results to as a csv')

    parser.add_argument('--mz_result_folder',
                        default=None,
                        help='path to mz folder with many test prediction outputs')

    args = parser.parse_args()
    if args.model == 'doc2vec':
        doc2vec_eval(args.datapath)
    elif args.model == 'word2vec':
        word2vec_eval(args.datapath, args.word_embedding_path)
    elif args.model == 'mz':
        mz_eval(args.mz_result_file)
    elif args.model == 'mz_folder':
        mz_eval_multiple(args.mz_result_folder)
    elif args.model == 'all':
        doc2vec_eval(args.datapath)
        word2vec_eval(args.datapath, args.word_embedding_path)
        if args.mz_result_file is not None:
            mz_eval(args.mz_result_file)
        elif args.mz_result_folder is not None:
            mz_eval_multiple(args.mz_result_folder)

    if args.result_save_path is not None:
        write_results_to_file(results_list, args.result_save_path)
