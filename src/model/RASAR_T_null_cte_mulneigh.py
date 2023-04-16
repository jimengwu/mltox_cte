import pandas as pd
from scipy.spatial.distance import jaccard, pdist, squareform, cdist
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from time import ctime
from tqdm import tqdm
import argparse

from helper_cte_model import *
import multiprocessing as mp
from sklearn.model_selection import train_test_split, ParameterSampler
import pickle
import os
import sys
from pebble import ProcessPool
from concurrent.futures import TimeoutError
import traceback

sys.modules["__main__"].__file__ = "ipython"
# db_invitro_matrix not defined in this code
invitro_matrices = None


def getArguments():
    parser = argparse.ArgumentParser(
        description="Find the best threshold for in vitro dataset \
                                     to make the invitro label fit with in vivo laebl."
    )
    parser.add_argument(
        "-i", "--input", dest="inputFile", help="input file position", required=True
    )
    parser.add_argument(
        "-n",
        "--neighbors",
        dest="neighbors",
        required=True,
        help="cloest neighbor number in rasar function",
        nargs="+",
        type=int,
    )
    parser.add_argument(
        "-e",
        "--encoding",
        dest="encoding",
        help="binary or multiclass (5 class)",
        default="binary",
    )
    parser.add_argument("-iv", "--invitro", dest="invitro", default="False")
    parser.add_argument(
        "-ni",
        "--niter",
        dest="niter",
        default=50,
        help="model iteration number to find the best hyperparameters",
        type=int,
    )
    parser.add_argument(
        "-il",
        "--invitro_label",
        dest="invitro_label",
        help="number, label",
        default="number",
    )
    parser.add_argument(
        "-dbi",
        "--db_invitro",
        dest="db_invitro",
        help="yes: add in vitro as other source for distance matrix, no: do not use in vitro as input, overlap: use in vitro as input feature",
        default="no",
    )
    parser.add_argument(
        "-wi",
        "--w_invitro",
        dest="w_invitro",
        help="own:in vitro alone as input  , \
            false:in vitro not as input ,\
            true:use in vitro and in vivo as input",
        default="False",
    )
    parser.add_argument("-ah", "--alpha_h", dest="hamming_alpha", default="logspace")
    parser.add_argument("-ap", "--alpha_p", dest="pubchem2d_alpha", default="logspace")
    parser.add_argument(
        "-r",
        "--repeat",
        dest="repeat",
        help="repeat time for different splitting method",
        default=20,
        type=int,
    )
    parser.add_argument("-t_ls", "--t_ls", dest="t_ls", default="median")

    parser.add_argument("-o", "--output", dest="outputFile", default="binary.txt")
    return parser.parse_args()


args = getArguments()


if __name__ == "__main__":

    categorical, conc_column = get_col_ls(args.invitro)

    if args.encoding == "binary":
        encoding = "binary"
        encoding_value = 1
    elif args.encoding == "multiclass":
        encoding = "multiclass"
        encoding_value = [0.1, 1, 10, 100]

    # loading data & splitting into train and test dataset
    print("loading dataset...", ctime())
    db_mortality = load_data(
        args.inputFile,
        encoding=encoding,
        categorical_columns=categorical,
        conc_column=conc_column,
        encoding_value=encoding_value,
    )

    print("finish loaded.", ctime())

    df_fishchem = db_mortality[["fish", "test_cas", conc_column]]

    record = []
    for repeat in range(args.repeat):

        filename = args.outputFile + "_log_{}.txt".format(repeat)

        grid_search = pd.DataFrame()

        # train valid dataset splitting

        trainvtest_idx, valid_idx = get_grouped_train_test_split(
            db_mortality[["fish", "test_cas", conc_column]],
            test_size=0.2,
            col_groups="test_cas",
            rand=repeat,
        )

        if list(valid_idx) in record:
            continue
        else:
            record.append(list(valid_idx))

        df_fishchem_tv = db_mortality[["fish", "test_cas", conc_column]].iloc[
            trainvtest_idx, :
        ]

        X = db_mortality.drop(columns=conc_column)

        X_traintest, X_valid, Y_trainvalid, Y_valid = get_train_test_data(
            db_mortality, trainvtest_idx, valid_idx, conc_column
        )
        count = 1
        if args.t_ls == "median":
            print(X_traintest["invitro_conc"].quantile([0.5]).values)
            threshold_ls = X_traintest["invitro_conc"].quantile([0.5]).values
        else:
            # threshold_ls = np.logspace(-1, 0.9, 20)
            threshold_ls = np.logspace(-5, 11, 5000)
            # threshold_ls = np.logspace(-1, 2, 30)

        for t in tqdm(threshold_ls):

            X_traintest = vitroconc_to_label(X_traintest, t)
            X_valid = vitroconc_to_label(X_valid, t)
            X = vitroconc_to_label(X, t)
            temp_grid = pd.DataFrame()
            temp_grid = pd.concat(
                [temp_grid, pd.DataFrame([t], columns=["threshold"]),], axis=1,
            )

            dict_acc = dataset_acc(X_traintest, X_valid, Y_trainvalid, Y_valid)

            temp_grid["train_dataset_accuracy"] = (
                str(round(dict_acc["train_acc"], 4))
                + " ("
                + str(dict_acc["train_correct"])
                + "/"
                + str(dict_acc["train_total"])
                + ")"
            )

            temp_grid["test_dataset_accuracy"] = (
                str(round(dict_acc["test_acc"], 4))
                + " ("
                + str(dict_acc["test_correct"])
                + "/"
                + str(dict_acc["test_total"])
                + ")"
            )

            # print(avg_accs, temp_grid)
            grid_search = pd.concat([grid_search, temp_grid])

        print("finished", ctime())
        # ----------------save the information into a file-------
        df2file(grid_search, args.outputFile + "_{}.txt".format(repeat))


# ------------------------------------------------------invitro + invivo to invivo(otv)----------------------------------------------------------
# binary & R unrelated
# python RASAR_mulneigh_bi_cte_T_null.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv -r 100 -n 1   -t_ls "tuned"  -dbi "overlap" -wi "own"   -il "label"   -o "T_models/T_null/repeat_own_label_new"
# python RASAR_mulneigh_bi_cte_T_null.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv -r 100 -n 1   -t_ls "median" -dbi "overlap" -wi "own"   -il "label"   -o "T_models/T_median/repeat_own_label"

# cd /local/wujimeng/code_jimeng/c_cte/GitHub/
# source activate rdkit-test
