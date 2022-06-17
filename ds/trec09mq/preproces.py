import pandas as pd

def prep_query(types, query):
    qe = pd.read_csv(f'{query}', encoding="ISO-8859-1", sep=':', index_col=False, names=['Topic','space', 'query'])
    qe_prep = qe.drop('space', axis=1)
    qt = pd.read_csv(f'{types}', encoding="ISO-8859-1", sep='\t')
    merge_query=pd.merge(qe_prep, qt, on='Topic', how='inner')
    merge_query.to_csv(r'./prep/09.mq.topics.20001-60000.prep' ,columns = ['Topic','query'],header=None, index=None,sep='\t', mode='w')

def prep_qrel(qrel):
    qr = pd.read_csv(f'{qrel}', encoding="ISO-8859-1", sep=' ',names = ['Topic','doc','score1','score2','score3'], index_col=False)
    qr.insert(1, 'space', "1")
    data = qr.drop('score2', axis=1)
    data1 = data.drop('score3', axis=1)
    data1.to_csv(r'./prep/prels.20001-60000.prep', header=None, index=None, sep='\t', mode='a')
    print(data1)
#09.mq.topics.20001-60000.prep
#prels.20001-60000.prep

if __name__ == "__main__":
    prep_qrel('./raw/prels.20001-60000')
    prep_query('./raw/queryclasses', './raw/09.mq.topics.20001-60000')

