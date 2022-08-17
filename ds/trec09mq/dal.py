import os, subprocess
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold


#s2i--> string to index
def load_refiners(file):
   # E = pd.read_csv(filename, header=0)
    r2i = {'nil': 0}
    i2r = {0: 'nil'}
    for idx, row in file.iterrows():
        r2i[row] = idx + 1
        i2r[idx + 1] = row
    return r2i, i2r

def load_ds(corpus, ranker, metric, r2i):
    Q = pd.read_csv(f'./../ds/{corpus}/topics.{corpus}.{ranker}.{metric}.dataset.csv', header=0, index_col='qid')
    all = pd.read_csv(f'./../ds/{corpus}/topics.{corpus}.{ranker}.{metric}.all.csv', header=0, index_col='qid')
    #Q['method.1'] = Q.apply(lambda r: r2i[r['method.1'] if not pd.isna(r['method.1']) else 'nil'], axis=1)
    #Q['method.1'] = Q.apply(lambda r: np.random.randint(2), axis=1)

    for idx, row in Q.iterrows():
        if pd.isna(row.abstractqueryexpansion):
            continue
        for i in range(1, 100 + 1):
            qcol = 'query.' + str(i)
            mcol = 'method.' + str(i)
            if (qcol not in Q.columns):
                break
            Q.loc[idx, mcol] = r2i[row[mcol]] if not pd.isna(row[mcol]) else r2i['nil']
            # check if the query string is a dict (for weighted expanders such as onfields)
            try:
                Q.loc[idx, qcol] = ' '.join(eval(row[qcol]).keys())
            except:
                pass
    all.columns = [r2i[col.replace(f'.{ranker}.{metric}', '')]
                   if col.find(f'.{ranker}.{metric}') > -1 and col.replace(f'.{ranker}.{metric}', '') in r2i.keys()
                   else col for col in all.columns]
    all.rename(columns={f'abstractqueryexpansion.{ranker}.{metric}': 0}, inplace=True)

    return Q, all

def create_evaluation_splits(n_sample, n_folds):
    train, test = train_test_split(np.arange(n_sample), train_size=0.7, test_size=0.3, random_state=0, shuffle=True)
    splits = dict()
    splits['test'] = test
    splits['folds'] = dict()
    skf = KFold(n_splits=n_folds, random_state=0, shuffle=True)
    #for k, (trainIdx, validIdx) in enumerate(skf.split(train)):
    for k, (trainIdx) in enumerate(skf.split(train)):
        splits['folds'][k] = dict()
        splits['folds'][k]['train'] = train[trainIdx]
        #splits['folds'][k]['valid'] = train[validIdx]

    return splits



from itertools import product
def merge_trec_topics(ds_path, corpus, ranker, metric):
    df = pd.DataFrame()
    if corpus == 'gov2':
        for r in ['4.701-750', '5.751-800', '6.801-850']:
            filename = f'{ds_path}/gov2/topics.terabyte0{r}.{ranker}.{metric}.all.csv'
            df = pd.concat([df, pd.read_csv(filename)], axis=0, ignore_index=True, sort=False)
        df.to_csv(f'{ds_path}/{corpus}/topics.gov2.{ranker}.{metric}.all.csv', index=False)
        pd.read_csv(f'{ds_path}/{corpus}/topics.gov2.701-850.{ranker}.{metric}.dataset.csv').to_csv(f'{ds_path}/{corpus}/topics.{corpus}.{ranker}.{metric}.dataset.csv', index=False)

    if corpus == 'clueweb09b':
        for r in ['1-50', '51-100', '101-150', '151-200']:
            filename = f'{ds_path}/clueweb09b/topics.web.{r}.{ranker}.{metric}.all.csv'
            df = pd.concat([df, pd.read_csv(filename)], axis=0, ignore_index=True, sort=False)
        df.to_csv(f'{ds_path}/{corpus}/topics.{corpus}.{ranker}.{metric}.all.csv', index=False)
        pd.read_csv(f'{ds_path}/{corpus}/topics.{corpus}.1-200.{ranker}.{metric}.dataset.csv').to_csv(f'{ds_path}/{corpus}/topics.{corpus}.{ranker}.{metric}.dataset.csv', index=False)

    if corpus == 'clueweb12b13':
        for r in ['201-250', '251-300']:
            filename = f'{ds_path}/clueweb12b13/topics.web.{r}.{ranker}.{metric}.all.csv'
            df = pd.concat([df, pd.read_csv(filename)], axis=0, ignore_index=True, sort=False)
        df.to_csv(f'{ds_path}/{corpus}/topics.{corpus}.{ranker}.{metric}.all.csv', index=False)
        pd.read_csv(f'{ds_path}/{corpus}/topics.{corpus}.201-300.{ranker}.{metric}.dataset.csv').to_csv(f'{ds_path}/{corpus}/topics.{corpus}.{ranker}.{metric}.dataset.csv', index=False)