import pandas as pd

def prep(query_types, queries, qrel):
    qe = pd.read_csv(queries, encoding="ISO-8859-1", sep=':', index_col=False, names=['Topic', ' ', 'query'], usecols=['Topic', 'query'])
    qt = pd.read_csv(query_types, encoding="ISO-8859-1", sep='\t')
    qr = pd.read_csv(qrel, encoding="ISO-8859-1", sep=' ', names=['Topic', 'doc', 'score1', 'score2', 'score3'], index_col=False, usecols=['Topic', 'doc', 'score1'])
    qet = pd.merge(qe, qt, on='Topic', how='inner')
    qrt = pd.merge(qr, qt, on='Topic', how='inner')

    qet.to_csv(r'./09.mq.topics.20001-60000.prep', columns=['Topic', 'query'], header=None, index=None, sep='\t')
    qrt.to_csv(r'./prels.20001-60000.prep', columns=['Topic', 'doc', 'score1'], header=None, index=None, sep='\t')

if __name__ == "__main__":
    prep('./queryclasses', './09.mq.topics.20001-60000', './prels.20001-60000')

