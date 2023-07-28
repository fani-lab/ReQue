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


def get_improved_data_all_dataset(source, corpora):
    metrics = ['bm25.map', 'bm25.recip_rank', 'qld.map', 'qld.recip_rank']
    for index, each_metric in enumerate(metrics):
        outfile_path = source + "results/" + f"{each_metric}.improved.queries.all.datasets.txt"
        with open(outfile_path, "w") as file:
            for corpus in corpora:
                file.write('<corpus> ' + corpus.split(".")[0] + '\n')
                infile_df = pd.read_csv(f'{source}{corpus.split(".")[0]}/topics.{corpus}.bm25.qld.map.recip_rank.all.csv')
                for column in range(6, len(infile_df.columns), 7):
                    # output language name
                    file.write('<lang> ' + str(infile_df.columns.tolist()[column + 1])[16:] + '\n')
                    for row in range(len(infile_df.iloc[:, column])):
                        if (infile_df.iloc[row, column + 3 + index] - infile_df.iloc[row, index + 2]) > 0:
                            # output format: <qid> <original_query> <original_query_eval> <backtranslation_Lang> <semsim> <backtranslation_Lang_eval> <subtraction(original - backtranslation)>
                            file.write(str(infile_df.iloc[row, 0]) +
                                       "\t" + str(infile_df.iloc[row, 1]) +
                                       "\t" + str(infile_df.iloc[row, index + 2]) +
                                       "\t" + str(infile_df.iloc[row, column + 1]) +
                                       "\t" + str(infile_df.iloc[row, column + 2]) +
                                       "\t" + str(infile_df.iloc[row, column + 3 + index]) +
                                       "\t" + str(infile_df.iloc[row, column + 3 + index] - infile_df.iloc[row, 2 + index]))
                            file.write("\n")


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
    files_in_directory = os.listdir(source + 'results/')
    file_paths_list = [os.path.join(source + 'results/', file) for file in files_in_directory if file.endswith('.improved.queries.all.datasets.txt') and 'analyze' not in file]
    for file in file_paths_list:
        analyze_improved_queries_all_dataset(source, infile_path=file)
    # plot_all_datasets(pd.read_csv(f"{source}analyze.bm25.qld.map.mrr.10lang.alldataset.csv"), outfile_path=source)
