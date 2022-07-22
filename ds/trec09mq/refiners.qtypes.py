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
    with open(output, 'wb') as f: pickle.dump(result, f)
    return result

def heatmap(matrix, output):
    with open(matrix, 'rb') as f: df = pickle.load(f)
    df=df.transpose()
    sns.heatmap(df, annot=True)
    plt.title("Distribution of query refiners in query types", fontsize=12)
    plt.savefig(output, bbox_inches='tight', dpi=100)
    plt.show()

def normalize_by_qtype(all,raw_result,queryclass,output):
    # load queryclass file and find the frequency
    # laod the raw_result
    # update the raw_result relative to the frequec
    # dump the update (./'pkl)
    # heat(update, output/ *png)
    qtypes = pd.read_csv(queryclass, sep='\t', usecols=['Topic', 'Class'])
    refiners = pd.read_csv(all, index_col=0, nrows=0).columns[3::3]
    qtypes_counts = Counter(qtypes['Class'])
    with open(raw_result, 'rb') as f: df = pickle.load(f)
    for qt in qtypes_counts:
        for r in refiners:
            df[qt][r]=(df[qt][r]*100)/qtypes_counts[qt]
    with open(output, 'wb') as f: pickle.dump(df, f)
    heatmap(output,"./normalized.frequency.png")



def normalize(raw_result,output):
    with open(raw_result, 'rb') as f: df = pickle.load(f)
    #n= (df - df.mean()) / df.std() #mean normalization
    df=(df-df.min())/(df.max()-df.min()) #min-max normalization
    with open(output, 'wb') as f: pickle.dump(df, f)
    heatmap(output, "./normalized.min_max.png")


if __name__ == "__main__":
    output = 'refiners.qtypes.pkl'
    output_norm_freq='normalized.frequency.pkl'
    output_norm_min_max='normalized.min_max.pkl'
    #result = get_refiners_by_qtypes('../../qe/output/trec09mq/topics.trec09mq.bm25.map.all.csv',
                                   # '../../qe/output/trec09mq/topics.trec09mq.bm25.map.dataset.csv',
                                    #'./queryclasses',
                                    #output)
    #heatmap(output, "./refiners.qtypes.png")
    normalize_by_qtype('../../qe/output/trec09mq/topics.trec09mq.bm25.map.all.csv',output,'./queryclasses',output_norm_freq)
    normalize(output,output_norm_min_max)