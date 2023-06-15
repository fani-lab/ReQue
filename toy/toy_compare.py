import re
import optparse
import numpy as np
import pandas as pd


def calculate_tValue_cohensD(map1, map2):
    # Calculate the mean and standard deviation
    mean_diff = map2 - map1
    std_diff = np.sqrt((map1 ** 2 + map2 ** 2) / 2)

    # Calculate Cohen's d
    cohens_d = mean_diff / std_diff

    n = 1  # Number of observations (in this case, each value is a single observation)
    std_error = std_diff / np.sqrt(n)
    t_value = mean_diff / std_error

    return t_value, cohens_d


def compare_mAP_each_row(infile_df, outfile_path, columns):
    # Header
    headers = list(infile_df.columns)
    headers.append('t_value')
    headers.append('cohens_d')
    headers.remove('index')
    headers = [x for x in headers if not re.match(r"^qid.*\d|\w$", x)]

    # Output file dataframe
    csv_result = pd.DataFrame(columns=headers)

    map_columns = [word for word in headers if word.endswith("map")]

    for index, row in infile_df.iterrows():

        # mAP for the two expanders
        map1 = row[[word for word in map_columns if word.startswith(columns.col1)][0]]
        map2 = row[[word for word in map_columns if word.startswith(columns.col2)][0]]

        # Check empty cells
        t_value, cohens_d = calculate_tValue_cohensD(map1=map1, map2=map2)

        # New row of the output file csv file
        new_line=[]
        new_line.append(row.qid)  # qid
        new_line.append(row[columns.col1])  # qid
        new_line.append(map1)  # col 1 mAP
        new_line.append(row[columns.col2])  # col 2 mAP
        new_line.append(map2)  # col 2 mAP
        new_line.append(t_value)  # t_value
        new_line.append(cohens_d)  # cohens_d

        csv_result.loc[len(csv_result)] = new_line

    # write dataFrame to SalesRecords CSV file
    output_file_name = f'compare.mAP.{columns.col1}.{columns.col2}.csv'
    csv_result.to_csv(outfile_path + output_file_name)


def compare_mAP_all_row(infile_df, outfile_path, columns):
    headers = list(infile_df.columns)
    map_columns = [word for word in headers if word.endswith("map")]
    map1_header = [word for word in map_columns if word.startswith(columns.col1)][0]
    map2_header = [word for word in map_columns if word.startswith(columns.col2)][0]

    infile_df.dropna(subset=[map1_header, map2_header])

    map1 = infile_df[[word for word in map_columns if word.startswith(columns.col1)][0]].tolist()
    map1 = [0 if str(item) == 'nan' else item for item in map1]
    map2 = infile_df[[word for word in map_columns if word.startswith(columns.col2)][0]].tolist()
    map2 = [0 if str(item) == 'nan' else item for item in map2]

    mean1 = np.mean(map1)
    mean2 = np.mean(map2)

    output_file_name = f'compare.mAP.{columns.col1}.{columns.col2}.txt'
    with open(outfile_path+output_file_name, 'w') as file:
        file.write(f'{columns.col1} mean of mAP is: {mean1}\n')
        file.write(f'{columns.col2} mean of mAP is: {mean2}\n')


def get_input():
    # Instantiate the parser
    parser = optparse.OptionParser()
    parser.add_option('--column1',
                      action="store", dest="col1",
                      help="query string", default="abstractqueryexpansion")
    parser.add_option('--column2',
                      action="store", dest="col2",
                      help="query string", default="backtranslation")

    options, args = parser.parse_args()

    return options


if __name__ == '__main__':
    columns = get_input()

    # Constant
    infile_path = '../qe/output/robust04/topics.robust04.bm25.map.all.csv'
    outfile_path = '../qe/output/compare/'

    infile_df = pd.read_csv(infile_path)
    infile_df = infile_df.reset_index()  # making sure indexes pair with number of rows

    compare_mAP_each_row(infile_df=infile_df, outfile_path=outfile_path, columns=columns)
    compare_mAP_all_row(infile_df=infile_df, outfile_path=outfile_path, columns=columns)




