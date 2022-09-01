from scipy import sparse
import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from lib import text_corpus as tc, utils

def extract_features(Q, feature_set=[]):#['basic', 'linguistic', 'w2v', 'bert']
    char_ngram_range = (1, 1)
    word_ngram_range = (1, 1)
    tc_Q = tc.TextCorpus(Q, char_ngram_range=char_ngram_range, word_ngram_range=word_ngram_range)

    q_features = sparse.csr_matrix(tc_Q.getLengthsByTerm()).transpose()

    if 'basic' in feature_set: q_features = sparse.csr_matrix(sparse.hstack((
        q_features,
        #tc_Q.getCharStat()[0], yandex words are ids
        tc_Q.getTermStat()[0],
        tc_Q.getTfIdF(),
        # tc_Q.getCharBM25(),
        # tc_Q.getTermBM25()

    )))
    if 'w2v' in feature_set:
        model = SentenceTransformer('average_word_embeddings_glove.6B.300d')
        sentence_embeddings = model.encode(Q.values)
        q_features = sparse.csr_matrix(sparse.hstack((q_features, sentence_embeddings)))
    if 'bert' in feature_set:
        model = SentenceTransformer('paraphrase-distilroberta-base-v2')
        sentence_embeddings = model.encode(Q.values)
        q_features = sparse.csr_matrix(sparse.hstack((q_features, sentence_embeddings)))

    return q_features


def extract_load_q_features(Q, y, feature_set, q_features_file=None):
    try:
        features_label = utils.load_sparse_csr(q_features_file)
        x = features_label[:, :-1]
        y = features_label[:, -1:]
        return x, y
    except:
        x = extract_features(Q, feature_set)
        y = sparse.csr_matrix(y, dtype=float).transpose()
        x_y = sparse.csr_matrix(sparse.hstack((x, y)))
        utils.save_sparse_csr(q_features_file, x_y)
        return x, y
