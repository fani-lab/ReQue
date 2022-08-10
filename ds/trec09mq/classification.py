from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import f1_score, roc_auc_score
import pandas as pd
import numpy as np
import dal
import extract_features as ef
import os, traceback, math, threading, time
from joblib import dump, load
import sys
import torch #just for topk(1)! should be replaced.
sys.path.extend(["../"])
import multiprocessing

def preprocess_data(dataset, qtypes):
    df_results= pd.read_csv(dataset, sep=',')
    #df_results.drop(df_results[df_results['star_model_count'] ==0].index, inplace=True) # drop rows with star_model_count=0
    qtypes = pd.read_csv(qtypes, sep='\t', usecols=['Topic', 'Class'])
    qtypes['Topic'] = qtypes['Topic'].astype(int)
    qtypes.rename(columns=({'Topic': 'qid'}), inplace=True)
    df = pd.merge(df_results, qtypes, on='qid', how='inner')
    return  df

def create_evaluation_splits(n_sample, n_folds):
    train, test = train_test_split(np.arange(n_sample), train_size=0.7, test_size=0.3, random_state=0, shuffle=True)
    splits = dict()
    splits['test'] = test
    splits['folds'] = dict()
    skf = KFold(n_splits=n_folds, random_state=0, shuffle=True)
    for k, (trainIdx, validIdx) in enumerate(skf.split(train)):
        splits['folds'][k] = dict()
        splits['folds'][k]['train'] = train[trainIdx]
        #splits['folds'][k]['valid'] = train[validIdx]

    return splits

def calc_metrics(y, y_, probs):
    metrics = dict()
    metrics['macro_auc_ovo'] = metrics['weighted_auc_ovo'] = metrics['macro_auc_ovr'] = metrics['weighted_auc_ovr'] = metrics['weighted_f1'] = None

    try:
        metrics['weighted_f1'] = f1_score(y_, y, average='weighted')
        metrics['macro_auc_ovo'] = roc_auc_score(y, probs, multi_class="ovo", average="macro")
        metrics['weighted_auc_ovo'] = roc_auc_score(y, probs, multi_class="ovo", average="weighted")
    except:
        print(traceback.format_exc())
        pass
    return metrics

def log_metrics(metrics, output):
    with open(output, 'w') as f:
        f.write('fold#,')
        header = 0
        for fold, values in metrics.items():
            if header == 0:
                for metric_name in metrics[fold].keys():
                    f.write(f'{metric_name},')
                f.write('\n')
                header = 1
            f.write(f'{fold},')
            for value in values.values():
                f.write(f'{value},')
            f.write('\n')

def learn_kf(X, y, model, splits, Q, output):
    metrics = {}
    for foldidx in splits['folds'].keys():
        X_train = X[splits['folds'][foldidx]['train'], :]
        y_train = y[splits['folds'][foldidx]['train']]
        # X_valid = X[splits['folds'][foldidx]['valid'], :]
        # y_valid = y[splits['folds'][foldidx]['valid']]


        model.fit(X_train, y_train)
        if not os.path.isdir(f'{output}'): os.makedirs(f'{output}')
        dump(model, f'{output}/{model.__class__.__name__}.train.fold{foldidx}.pkl')
        # probs = model.predict_proba(X_valid)

        # _, idxes = torch.as_tensor(model.predict_proba(X_valid)).topk(1)
        # y_pred = [model.classes_[i] for i in idxes]
        # metrics[foldidx] = calc_metrics(y_valid, y_pred, probs.max(axis=1))
        #df = pd.DataFrame(np.asarray([Q.loc[Q.index[splits['folds'][foldidx]['valid']]]['Original Query'], splits['folds'][foldidx]['valid'], y_valid, y_pred]).transpose(), columns=['List of Terms', 'idx', 'true', 'pred'], dtype=object)
        df.to_csv(f'{output}/{model.__class__.__name__}.valid{foldidx}.pred.csv', index=False)

    log_metrics(metrics, f'{output}/{model.__class__.__name__}.valid.log')

def test(X, y, model, splits, Q, output):
    metrics = dict()
    X_test = X[splits['test'], :]
    y_test = y[splits['test']]

    all_train = np.concatenate([splits['folds'][0]['train'], splits['folds'][0]['valid']])
    X_train = X[all_train, :]
    y_train = y[all_train]
    model.fit(X_train, y_train)
    if not os.path.isdir(f'{output}'): os.makedirs(f'{output}')
    dump(model, f'{output}/{model.__class__.__name__}.train.fold_.pkl')
    probs = model.predict_proba(X_test)
    _, idxes = torch.as_tensor(model.predict_proba(X_test)).topk(1)
    y_pred = [model.classes_[i] for i in idxes]
    metrics['_'] = calc_metrics(y_test, y_pred, probs.max(axis=1))
    df = pd.DataFrame(np.asarray([Q.loc[Q.index[splits['test']]]['Original Query'], splits['test'],y_test, y_pred]).transpose(), columns=['List of Terms', 'idx', 'true', 'pred'], dtype=object)
    df.to_csv(f'{output}/{model.__class__.__name__}.test_.pred.csv', index=False)

    for foldidx in splits['folds'].keys():
        model = load(f'{output}/{model.__class__.__name__}.train.fold{foldidx}.pkl')
        probs = model.predict_proba(X_test)
        _, idxes = torch.as_tensor(model.predict_proba(X_test)).topk(1)
        y_pred = [model.classes_[i] for i in idxes]
        metrics[foldidx] = calc_metrics(y_test, y_pred, probs.max(axis=1))
        df = pd.DataFrame(np.asarray([Q.loc[Q.index[splits['test']]]['Original Query'], splits['test'],y_test, y_pred]).transpose(), columns=['List of Terms', 'idx', 'true', 'pred'], dtype=object)
        df.to_csv(f'{output}/{model.__class__.__name__}.test{foldidx}.pred.csv', index=False)

    log_metrics(metrics, f'{output}/{model.__class__.__name__}.test.log')

def main(splits, Q, y,feature_set, path, cmd=['prep', 'train', 'eval', 'test']):
    q_features, q_labels = ef.extract_load_q_features(Q, y,feature_set, q_features_file=path)
    print(q_features)
#     if 'userid' in feature_set:
#         userID_col = np.array(Q['UserID'])[:,None]
#         q_features = sparse.hstack([userID_col, q_features]).tocsr()
#
    cores = multiprocessing.cpu_count()
    models = [LogisticRegression(n_jobs=cores, max_iter=100, random_state=0),
              RandomForestClassifier(n_jobs=cores, n_estimators=100, max_depth=10, random_state=0),  #
              # XGBClassifier(n_jobs=cores, n_estimators=1000, max_depth=15, learning_rate=0.1, random_state=random_seed),
              # MLP (word embdedding in input, n_refiners in output)
              # multilabel
              ]

    output = f'{path}.qf'

    for model in models:
        if 'train' in cmd: learn_kf(q_features, q_labels.toarray().ravel(), model, splits=splits, Q=Q, output=output)
       # if 'test' in cmd: test(q_features, q_labels.toarray().ravel(), model, splits=splits, Q=Q, output=output)


def con_r2i(df,all):
    df_all=pd.read_csv(all, encoding="ISO-8859-1", sep=',')
    R = pd.read_csv(all, encoding="ISO-8859-1", sep=',').columns[4:len(df_all.columns):3]
    r2i = {}
    r= df['method.1'].unique()
    print(R)
    for i in range(len(r)):
        r2i[r[i]]=i+1
    for i in range(len(df)):
        # replace refinement with index
        df['method.1'][i]=r2i[df['method.1'][i]]
        #x=r2i[df['method.1'][i]]
        #df['method.1'] = df['method.1'].replace([df['method.1'][i]], int(x))
       # df.set_value('method.1', i, r2i[df['method.1'][i]])
        #f.loc[df['method.1'] == 'n', col] = 0
        #print(col)
        #x=r2i[df['method.1'][i]]
        #df.loc[:, 'method.1'][i]=
    return df

def con_t2i(df):
    t2i = {}
    t= df['Class'].unique()
    # print(t)
    for i in range(len(t)):
        t2i[t[i]]=i+1
    # print(t2i)
    for i in range(len(df)):
        # replace refinement with index
        df['Class'][i]=t2i[df['Class'][i]]
    #print(df['Class'])
    # the df contain both refinemnets and qtypes indexes
    return df

if __name__ == "__main__":
    df_results=preprocess_data('../../qe/output/trec09mq/topics.trec09mq.bm25.map.dataset.csv','./queryclasses')
    splits = create_evaluation_splits(len(df_results), 5)
    feature_sets = [['basic'], ['userid', 'basic']]
    con_r2i(df_results,'../../qe/output/trec09mq/topics.trec09mq.bm25.map.all.csv')
    #the df contain both refinemnets and qtypes indexes
    df=con_t2i(con_r2i(df_results,'../../qe/output/trec09mq/topics.trec09mq.bm25.map.all.csv'))
    #feature_sets = [['w2v'], ['bert'], ['w2v','bert'], ['userid', 'w2v'], ['userid', 'bert'], ['userid', 'w2v','bert']]
    feature_sets = [['bert'], ['userid', 'bert']]
    for feature_set in feature_sets:
        feature_set_str = '.'.join(feature_set)
        main(splits, df['abstractqueryexpansion'], df['method.1'],feature_set, f'./results.npz')

    # try:
    #     features_label = utils.load_sparse_csr(q_features_file)
    #     x=features_label[:, :-1]
    #     y=features_label[:, -1:]
    # except:
    #     x = extract_features(df_results['abstractqueryexpansion'], feature_set)
    #     x_y = sparse.csr_matrix(sparse.hstack((x, df_results["method.1"])))
    #     utils.save_sparse_csr(q_features_file, q_features_label)


    # if x = x + qtype

    #print(r2i_x,'\n', i2r_x)
   ##feature_set = ['basic']
    #main(splits, Q, r2i_x,feature_set, f'./results')
    # for feature_set in feature_sets:
    #     feature_set_str = '.'.join(feature_set)
    #     main(splits, Q, feature_set, f'../../output/yandex/yandex.{feature_set_str}')


