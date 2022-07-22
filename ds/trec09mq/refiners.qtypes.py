import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

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

if __name__ == "__main__":
    output = 'refiners.qtypes.pkl'
    result = get_refiners_by_qtypes('../../qe/output/trec09mq/topics.trec09mq.bm25.map.all.csv',
                                    '../../qe/output/trec09mq/topics.trec09mq.bm25.map.dataset.csv',
                                    './queryclasses',
                                    output)
    heatmap(output, "./refiners.qtypes.png")