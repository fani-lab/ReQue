import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

def get_refiners_by_qtypes(all, dataset, qtypes, output):

    refiners = pd.read_csv(all, index_col=0, nrows=0).columns[3::3]
    q_ = pd.read_csv(dataset, sep=',')
    qtypes = pd.read_csv(qtypes, sep='\t', usecols=['Topic', 'Class'])
    qtypes['Topic'] = qtypes['Topic'].astype(int)
    qtypes.rename(columns=({'Topic': 'qid'}), inplace=True)

    q_ = pd.merge(q_, qtypes, on='qid', how='inner')
    qtypes = qtypes['Class'].unique()

    result = dict()
    for qt in qtypes:
        result[qt] = dict()
        for r in refiners: result[qt][r] = 0

    for index, row in q_.iterrows():
        for i in range(1, row['star_model_count'] * 3, 3): result[row["Class"]][str(row[i + 3])] += 1

    result = pd.DataFrame.from_dict(result)
    with open(f'{output}refiners.qtypes.pkl', 'wb') as f: pickle.dump(result, f)
    heatmap(result, f"{output}refiners.qtypes.png")
    return result

def heatmap(matrix, output, anot=False):
    df=matrix.transpose()
    sns.heatmap(df, annot=anot)
    plt.title("Distribution of query refiners in query types", fontsize=12)
    plt.savefig(output, bbox_inches='tight', dpi=100)
    plt.show()

def normalize_by_qtype(raw_result, queryclass, output):
    qtypes = pd.read_csv(queryclass, sep='\t', usecols=['Topic', 'Class'])
    qtypes_counts = Counter(qtypes['Class'])

    with open(raw_result, 'rb') as f: df = pickle.load(f)
    df = df.astype(float, copy=False)
    refiners = df.index.values
    for qt in qtypes_counts:
        for r in refiners:
            df[qt][r] = df[qt][r] / qtypes_counts[qt]
    with open(f"{output}refiners.qtypes.norm.freq.pkl", 'wb') as f: pickle.dump(df, f)
    heatmap(df, f"{output}refiners.qtypes.norm.freq.png")
    return df

def normalize(raw_result,output):
    with open(raw_result, 'rb') as f: df = pickle.load(f)
    df = df.astype(float, copy=False)
    #n= (df - df.mean()) / df.std() #mean normalization
    df = (df - df.min().min()) / (df.max().max() - df.min().min()) #min-max normalization
    with open(f"{output}refiners.qtypes.norm.minmax.pkl", 'wb') as f: pickle.dump(df, f)
    heatmap(df, f"{output}refiners.qtypes.norm.minmax.png")
    return df

if __name__ == "__main__":
    result = get_refiners_by_qtypes('../../qe/output/trec09mq/topics.trec09mq.bm25.map.all.csv',
                                    '../../qe/output/trec09mq/topics.trec09mq.bm25.map.dataset.csv',
                                    './queryclasses',
                                    './')
    normalize_by_qtype('./refiners.qtypes.pkl', './queryclasses', './')
    normalize('./refiners.qtypes.pkl', './')