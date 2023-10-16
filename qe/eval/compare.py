import re
import optparse
import numpy as np
import pandas as pd
from cmn import param
import matplotlib.pyplot as plt

import os
import sys
sys.path.extend(['../qe/output'])


def calculate_tvalue_cohensd(map1, map2):
    """
        Calculate t-value and cohens-d for two maps.

        This function takes two maps and calculates t-value and cohens-d.

        Parameters:
            map1 (float): Map of the first expander
            map2(float): Map of the second expander
    """
    # Calculate the mean and standard deviation
    mean_diff = map2 - map1
    std_diff = np.sqrt((map1 ** 2 + map2 ** 2) / 2)

    if std_diff == 0:
        cohens_d = 0
        t_value = 0
    else:
        # Calculate Cohen's d
        cohens_d = mean_diff / std_diff

        n = 1  # Number of observations (in this case, each value is a single observation)
        std_error = std_diff / np.sqrt(n)
        t_value = mean_diff / std_error

    return t_value, cohens_d


def compare_map_each_row(infile_df, outfile_path, col1, col2):
    """
        Calculate t-value and cohens-d for two expanders' maps.

        This function takes the name of the expanders and compare their maps by calculating t-value and cohens-d.

        Parameters:
            infile_df (dataframe): Dataframe of all expanders and their map.
            outfile_path(str): Path of the output file
            col1(str): Name the first expander
            col2(str): Name the second expander
    """

    # Initialize header
    headers = ['qid']
    col2_list = []
    for word in list(infile_df.columns):
        if word.startswith(col2) and word.endswith("map"):
            headers.append(word)
            headers.append('t-value')
            headers.append('cohens-d')
            col2_list.append(word)
        elif word.startswith(col1) and word.endswith("map"):
            # Make sure the main col that wants to do the comparison with, is placed at the second col.
            # The first column that match the col1 input is considered for main col.
            headers.insert(1, word)
            col1 = word

    # Output csv file
    csv_result = pd.DataFrame(columns=headers)

    for index, row in infile_df.iterrows():
        # Initialize new line
        # If a list of names, starting with col1, is returned, only the first one is considered!
        map1 = 0 if str(row[col1]) == 'nan' else float(row[col1])
        new_line = [row.qid, map1]

        for index, map2 in enumerate(col2_list):
            m2 = 0 if str(row[map2]) == 'nan' else float(row[map2])
            t_value, cohens_d = calculate_tvalue_cohensd(map1=map1, map2=m2)

            # New row of the output file csv file
            new_line.append(row[map2])  # col 2 mAP
            new_line.append(t_value)  # t_value
            new_line.append(cohens_d)  # cohens_d

        csv_result.loc[len(csv_result)] = new_line

    # write dataFrame to a CSV file
    csv_result.to_csv(outfile_path + '/results/' + 'compare.row.map.csv')


def plot_all_datasets(infile_df, outfile_path):
    """
        Plot and save the results of all expanders per IR and evaluation metric.

        This function takes a csv file with headers as ['Dataset', 'Languages', 'IR Metric', 'Evaluation Metric']

        Parameters:
            infile_df (dataframe): Dataframe of input file with the above headers.
    """
    ir_metrics = infile_df['IR Metric'].unique()
    evaluation = infile_df['Evaluation Metric'].unique()
    datasets = infile_df['Dataset'].unique()
    colors = ['#b366ff', '#3399ff', '#00e673', '#ffcc00', '#ff751a']

    for each_metric in ir_metrics:
        for each_eval in evaluation:
            index = 0
            for each_dataset in datasets:
                df_selected = infile_df[(infile_df['IR Metric'] == each_metric) & (infile_df['Evaluation Metric'] == each_eval) & (infile_df['Dataset'] == each_dataset)]
                plt.plot(df_selected['Languages'].to_list(), df_selected['Results'].to_list(), color=colors[index], label=each_dataset)
                plt.xticks(rotation=90)
                plt.grid(True)
                index += 1
            plt.xlabel("Languages")
            plt.ylabel("Results")
            plt.title(f'{each_metric} & {each_eval} on all datasets!')
            plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            plt.tight_layout()
            plt.savefig(f'{outfile_path}/results/{each_metric}.{each_eval}.all.datasets.jpg')
            plt.show()


def analyze_improved_queries_all_language(infile_path):
    """
        Count all the improved queries per language.
        Calculate the average of map and semsim per language.
        This function takes the path of a txt file and read the tags and generate a dictionary.

        Parameters:
            infile_path (str): Path of the txt file with generated tags from get_improved_data_all_dataset() function.
        Returns:
            Returns a dictionary of a dictionary of the improved query. [key:corpus][key:language][value(list):number of improved queries, avg(semsim), avg(map)]
    """
    corpus, lang = "", ""
    count, semsim, metric = 0, 0, 0
    qid = set()
    dict_lang = dict()
    dict_all_improved = dict()
    with open(infile_path, 'r') as infile:
        for line in infile:
            if ('<corpus>' in line or '<lang>' in line) and count != 0:
                dict_lang[corpus][lang] = [count, semsim/count, metric/count]
            if '<corpus>' in line and count != 0:
                dict_all_improved[corpus] = len(qid)
            if '<corpus>' in line:
                corpus = line[line.index('>') + 1:].lower().strip()
                dict_lang[corpus] = dict()
                qid = set()
            elif '<lang>' in line:
                lang = line[line.index('>') + 1:].lower().strip()
                count = 0
                semsim = 0
                metric = 0
            elif line: # if the starts with qid
                count += 1
                semsim += float(line.split('\t')[4])
                metric += float(line.split('\t')[5])
                qid.add(str(line.split('\t')[0]))
    dict_lang[corpus][lang] = [count, semsim / count, metric / count]
    dict_all_improved[corpus] = len(qid)
    return dict_lang, dict_all_improved


def analyze_improved_queries_all_dataset(source, infile_path):
    """
        Count all the improved queries in a dataset.
        Get the number of improved queries per language from analyze_improved_queries_all_language() function.

        This function takes a csv file path which has all the improved queries of a dataset.

        Parameters:
            source (str): Path of the source file.
            dict_lang (dict): Dictionary of a dictionary of the improved query.
    """
    # Constant
    corpora = ['robust04', 'dbpedia', 'antique', 'clueweb09b.1-200', 'gov2.701-850']
    metric = (infile_path.split("/")[-1]).split('.')[:2]
    outfile_path = source + '/results/' + f'analyze.{".".join(metric)}.improved.queries.all.datasets.txt'
    dict_lang, improved = analyze_improved_queries_all_language(infile_path)

    with open(outfile_path, 'w') as file:
        for i in corpora:
            # Read the stored results
            dataset_csv_path = f'{source}{i.split(".")[0]}/topics.{i}.bm25.qld.map.recip_rank.all.csv'
            infile_df = pd.read_csv(dataset_csv_path)
            infile_df = infile_df.reset_index()  # making sure indexes pair with number of rows
            all_queries = len(infile_df)
            # Write to a file
            # Format: <Number of all queries> <All improved queries> <ir_eval_metric>
            file.write(f'{i}'
                       f'\t{all_queries}'
                       f'\t{(improved[i.split(".")[0]] / all_queries) * 100}'
                       # f'\t{improved[i.split(".")[0]]}'
                       f'\t{"_".join(metric)}')
            for each_lang in dict_lang[i.split(".")[0]].keys():
                # Percentage of improved queries from all queries
                # Format: <language> <average.semsim> <average.ir_eval_metric>
                file.write(f'\t{each_lang}'
                           f'\t{(dict_lang[i.split(".")[0]][each_lang][0]/all_queries)*100} % '
                           f'\t{dict_lang[i.split(".")[0]][each_lang][1]}'
                           f'\t{dict_lang[i.split(".")[0]][each_lang][2]}')
            file.write('\n')


def draw_similarity_chart(source):
    infile_df = pd.read_csv(f'{source}results/all.queries.all.datasets.csv')
    fig, ax = plt.subplots()
    infile_df = infile_df.fillna(0)
    colors = ["#FF573300",  "#FFC30000",  "#FFDC0000",  "#C2FF3300",  "#00FFA700",  "#0082FF00",  "#7800FF00",  "#FF00E500",  "#FF007700",  "#00FFCC00",  "#0082FF00"]

    original_terms = list(infile_df[infile_df['metrics'] == 'bm25.map'][infile_df['lang'] == 'fra_latn']['original_query'])
    unique_languages = list(set(infile_df['lang']))
    for index, lang in enumerate(unique_languages):
        q_qprim = []
        for i, ot in enumerate(original_terms):
            q_qprim.append(list(infile_df[infile_df['original_query'] == ot][infile_df['lang'] == lang][infile_df['metrics'] == 'bm25.map']['semsim'])[0])
        ax.hist(q_qprim, label=f'{lang}', range=(0, 1), alpha=0.5, color=colors[index], edgecolor='black')

    ax.set_xlabel('#tokens', fontsize=20, labelpad=10)
    ax.set_ylabel('count', fontsize=20, labelpad=10)
    ax.legend(loc='upper right', bbox_to_anchor=(1.72, 1.2), fontsize=20)
    plt.xticks(rotation=45, ha='right', fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid()
    plt.savefig(f'{source}results/semsim_histogram.png', dpi=100, bbox_inches='tight')
    plt.show()


def draw_count_chart(source):
    infile_df = pd.read_csv(f'{source}results/all.queries.all.datasets.csv')
    new_df = pd.DataFrame()
    # Create a figure and axis
    fig, ax = plt.subplots()

    colors = ["#FF573300",  "#FFC30000",  "#FFDC0000",  "#C2FF3300",  "#00FFA700",  "#0082FF00",  "#7800FF00",  "#FF00E500",  "#FF007700",  "#00FFCC00",  "#0082FF00"]

    # Extract data for plotting
    original_count = list(map(count_words, list(infile_df[infile_df['metrics'] == 'bm25.map'][infile_df['lang'] == 'fra_latn']['original_query'])))
    new_df['o_count'] = original_count
    ax.hist(original_count, label='original_query', alpha=0.4, color=colors[0], edgecolor=colors[0])
    langs = infile_df['lang']

    unique_languages = list(set(langs))
    # colors = plt.cm.get_cmap('tab20', len(unique_languages))
    for index, lang in enumerate(unique_languages):
        backtranslated_count = list(map(count_words, list(infile_df[infile_df['lang'] == lang][infile_df['metrics'] == 'bm25.map']['backtranslated_query'])))
        new_df[f'b_count_{lang}'] = backtranslated_count
        ax.hist(backtranslated_count, label=f'{lang}', range=(0, 20), alpha=0.5, color=colors[index+1], edgecolor='black')

    ax.set_xlabel('#tokens', fontsize=20, labelpad=10)
    ax.set_ylabel('count', fontsize=20, labelpad=10)
    ax.legend(loc='upper right', bbox_to_anchor=(1.72, 1.2), fontsize=20)
    plt.xticks(rotation=45, ha='right', fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid()
    plt.savefig(f'{source}results/count_histogram.png', dpi=100, bbox_inches='tight')
    plt.show()
    new_df.to_csv(source + "results/count_tokens.csv")


def count_words(term):
    if type(term) == float: return 0
    return len(term.split())


def count_word_differences(original, backtranslated):
    words1 = original.split()
    words2 = backtranslated.split()
    return abs(len(words2) - len(words1))


def get_improved_data_all_dataset(source, corpora):
    metrics = ['bm25.map', 'bm25.recip_rank', 'qld.map', 'qld.recip_rank']
    csv_result = pd.DataFrame(columns=['corpus', 'qid', 'original_query', 'original_query_eval', 'metrics', 'lang', 'backtranslated_query', 'semsim', 'backtranslated_query_eval', 'original-backtranslated'])
    outfile_path = source + "results/" + f"improved.queries.all.datasets.csv"
    # outfile_path = source + "results/improved.queries.all.datasets.csv"
    for index, each_metric in enumerate(metrics):
        for corpus in corpora:
            infile_df = pd.read_csv(f'{source}{corpus.split(".")[0]}/topics.{corpus}.bm25.qld.map.recip_rank.all.csv')
            for column in range(6, len(infile_df.columns), 7):
                for row in range(len(infile_df.iloc[:, column])):
                    # Accept only improved queries
                    if (infile_df.iloc[row, column + 3 + index] - infile_df.iloc[row, index + 2]) > 0:
                    # output format: 'corpus', 'qid', 'original_query', 'original_query_eval', 'metrics, 'lang', 'backtranslated_query', 'semsim', 'backtranslated_query_eval', 'original-backtranslated'
                        new_line = []
                        new_line.append(corpus.split(".")[0])
                        new_line.append(str(infile_df.iloc[row, 0]))
                        new_line.append(str(infile_df.iloc[row, 1]))
                        new_line.append(str(infile_df.iloc[row, index + 2]))
                        new_line.append(str(each_metric))

                        new_line.append(str(infile_df.columns.tolist()[column + 1])[16:])
                        new_line.append(str(infile_df.iloc[row, column + 1]))
                        new_line.append(str(infile_df.iloc[row, column + 2]))
                        new_line.append(str(infile_df.iloc[row, column + 3 + index]))
                        new_line.append(str(infile_df.iloc[row, column + 3 + index] - infile_df.iloc[row, 2 + index]))

                    csv_result.loc[len(csv_result)] = new_line

    csv_result.to_csv(outfile_path)


def compare_expanders(source, corpora):
    metrics = ['bm25.map', 'bm25.recip_rank', 'qld.map', 'qld.recip_rank']
    outfile_path = source + "results/" + f"all.queries.all.datasets.all.expanders.csv"
    compare_result = pd.DataFrame(columns=['corpus', 'Expander', 'avg.bm25.map', 'avg.bm25.recip_rank', 'avg.qld.map', 'avg.qld.recip_rank', '#.bm25.map', 'delta.metric.bm25.map', '#.bm25.recip_rank', 'delta.metric.bm25.recip_rank', '#.qld.map', 'delta.metric.qld.map', '#.qld.recip_rank', 'delta.metric.qld.recip_rank'])
    q_star=dict()
    for corpus in corpora:
        q_star[corpus]=dict()
        # Calculate averages
        infile_df = pd.read_csv(f'{source}{corpus.split(".")[0]}/topics.{corpus}.bm25.qld.map.recip_rank.all.csv')
        new_line = []
        for column in range(6, len(infile_df.columns), 6):
            new_line.append(corpus)                                       # corpus
            new_line.append(str(infile_df.columns.tolist()[column + 1]))  # Expander's name
            new_line.append(str(infile_df.iloc[:, column + 2].mean()))    # Average bm25.map
            new_line.append(str(infile_df.iloc[:, column + 3].mean()))    # Average bm25.recip_rank
            new_line.append(str(infile_df.iloc[:, column + 4].mean()))    # Average qld.map
            new_line.append(str(infile_df.iloc[:, column + 5].mean()))    # Average qld.recip_rank


            q_star[corpus][str(infile_df.columns.tolist()[column + 1])] = [0,0,0,0,0,0,0,0]
            for row in range(len(infile_df.iloc[:, column])):
                # '#.bm25.map', 'delta.metric.bm25.map'
                if (infile_df.iloc[row, column + 2] - infile_df.iloc[row, 2]) > 0:
                    q_star[corpus][str(infile_df.columns.tolist()[column + 1])][0] += 1
                    q_star[corpus][str(infile_df.columns.tolist()[column + 1])][1] += infile_df.iloc[row, column + 2] - infile_df.iloc[row, 2]
                # '#.bm25.recip_rank', 'delta.metric.bm25.recip_rank'
                if (infile_df.iloc[row, column + 3] - infile_df.iloc[row, 3]) > 0:
                    q_star[corpus][str(infile_df.columns.tolist()[column + 1])][2] += 1
                    q_star[corpus][str(infile_df.columns.tolist()[column + 1])][3] += infile_df.iloc[row, column + 3] - infile_df.iloc[row, 3]
                # '#.qld.map', 'delta.metric.qld.map'
                if (infile_df.iloc[row, column + 4] - infile_df.iloc[row, 4]) > 0:
                    q_star[corpus][str(infile_df.columns.tolist()[column + 1])][4] += 1
                    q_star[corpus][str(infile_df.columns.tolist()[column + 1])][5] += infile_df.iloc[row, column + 4] - infile_df.iloc[row, 4]
                # '#.qld.recip_rank', 'delta.metric.qld.recip_rank'
                if (infile_df.iloc[row, column + 5] - infile_df.iloc[row, 5]) > 0:
                    q_star[corpus][str(infile_df.columns.tolist()[column + 1])][6] += 1
                    q_star[corpus][str(infile_df.columns.tolist()[column + 1])][7] += infile_df.iloc[row, column + 5] - infile_df.iloc[row, 5]

            new_line.append(str(q_star[corpus][str(infile_df.columns.tolist()[column + 1])][0]))  # '#.bm25.map'
            new_line.append(str(q_star[corpus][str(infile_df.columns.tolist()[column + 1])][1]))  # 'delta.metric.bm25.map'
            new_line.append(str(q_star[corpus][str(infile_df.columns.tolist()[column + 1])][2]))  # '#.bm25.recip_rank
            new_line.append(str(q_star[corpus][str(infile_df.columns.tolist()[column + 1])][3]))  # delta.metric.bm25.recip_rank'
            new_line.append(str(q_star[corpus][str(infile_df.columns.tolist()[column + 1])][4]))  # '#.qld.map'
            new_line.append(str(q_star[corpus][str(infile_df.columns.tolist()[column + 1])][5]))  # 'delta.metric.qld.map'
            new_line.append(str(q_star[corpus][str(infile_df.columns.tolist()[column + 1])][6]))  # '#.qld.recip_rank'
            new_line.append(str(q_star[corpus][str(infile_df.columns.tolist()[column + 1])][7]))  # 'delta.metric.qld.recip_rank'

            compare_result.loc[len(compare_result)] = new_line

    compare_result.to_csv(outfile_path)





def get_input():
    # Instantiate the parser
    parser = optparse.OptionParser()
    parser.add_option('--column1',
                      action="store", dest="col1",
                      help="Query string, the main column you want to do the comparison",
                      default="abstractqueryexpansion")
    parser.add_option('--column2',
                      action="store", dest="col2",
                      help="Query string, Name of the expander or column(s) (this name can refer to multiple columns) "
                           "with want to compare to the first column",
                      default="backtranslation")
    options, args = parser.parse_args()
    return options


if __name__ == '__main__':
    corpora = ['robust04', 'dbpedia', 'antique', 'clueweb09b.1-200', 'gov2.701-850']
    source = '../qe/output/'
    columns = get_input()

    # for corpus in corpora:
    #     dataset_csv_path = f'{source}{corpus.split(".")[0]}/topics.{corpus}.bm25.qld.map.recip_rank.all.csv'
    #     compare_map_each_row(infile_df=pd.read_csv(dataset_csv_path), outfile_path=source, col1=columns.col1, col2=columns.col2)

    get_improved_data_all_dataset(source=source, corpora=corpora)
    # compare_expanders(source=source, corpora=corpora)
    # draw_count_chart(source=source)
    # draw_similarity_chart(source=source)
    # files_in_directory = os.listdir(source + 'results/')
    # file_paths_list = [os.path.join(source + 'results/', file) for file in files_in_directory if file.endswith('.improved.queries.all.datasets.txt') and 'analyze' not in file]
    # for file in file_paths_list:
    #     analyze_improved_queries_all_dataset(source, infile_path=file)
    # plot_all_datasets(pd.read_csv(f"{source}analyze.bm25.qld.map.mrr.10lang.alldataset.csv"), outfile_path=source)
