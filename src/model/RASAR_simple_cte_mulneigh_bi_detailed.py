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
import logging

# this file are used to test the rnighbor number's effect in simple RASAR function. \
# different straitifed way are also been looked. during the cross validation process, first we \
#searchyed the best alphas combination, and then used the found alphas to excplore the \
#hyperparameters range.  Since the stratified spliiting method might also influence the \
#performance of the model, we set a random seed "repeat "to identify each stratified time. \
# and the performance was recorded. 

sys.modules["__main__"].__file__ = "ipython"


def getArguments():
    parser = argparse.ArgumentParser(description="Running simple RASAR with different neighbor number \
                                     for stratified splitted datasets.")
    parser.add_argument("-i", "--input", dest="inputFile", required=True)
    parser.add_argument("-ah", "--alpha_h", dest="hamming_alpha", default="logspace")
    parser.add_argument("-ap", "--alpha_p", dest="pubchem2d_alpha", default="logspace")
    parser.add_argument("-dbi", "--db_invitro", dest="db_invitro", default="noinvitro")
    parser.add_argument("-e", "--encoding", dest="encoding", default="binary")
    parser.add_argument(
    "-il", "--invitro_label", dest="invitro_label", default="number"
    )
    parser.add_argument("-iv", "--invitro_type", dest="invitro_type", default="False")
    parser.add_argument(
    "-m", "--model_type", help="model: logistic regression, random forest", default="rf",
    )
    parser.add_argument(
        "-n", "--neighbors", dest="neighbors", required=True, nargs="+", type=int
    )


    parser.add_argument("-ni", "--niter", dest="niter", default=5, type=int)
    parser.add_argument("-r", "--repeat", dest="repeat", default=20, type=int)

    parser.add_argument("-wi", "--w_invitro", dest="w_invitro", default="False")


    parser.add_argument("-o", "--output", dest="outputFile", default="binary.txt")
    return parser.parse_args()


args = getArguments()


def s_rasar_func(
    train_index,
    test_index,
    dist_matr_train,
    dist_matr_test,
    y_train,
    y_test,
    X,
    n_neighbors,
    ah,
    ap,
    matrices_invitro,
    max_euc,
    num,
    model,
):

    if args.w_invitro == "own":
        train_rf = pd.DataFrame()
        test_rf = pd.DataFrame()
    else:
        train_rf, test_rf = cal_s_rasar(
            dist_matr_train,
            dist_matr_test,
            y_train,
            n_neighbors,
            args.encoding,
        )

    if args.w_invitro != "False":
        if str(args.db_invitro) == "overlap":
            train_rf = get_vitroinfo(train_rf, X, train_index, args.invitro_label)
            test_rf = get_vitroinfo(test_rf, X, test_index, args.invitro_label)
        else:
            db_invitro_matrix_normed = pd.DataFrame(
                ah * matrices_invitro["hamming"]
                + ap * matrices_invitro["pubchem"]
                + pd.DataFrame(matrices_invitro["euc"]).divide(max_euc).values
            )
            train_rf = find_nearest_vitro(
                train_rf,
                args.db_invitro,
                db_invitro_matrix_normed,
                train_index,
                args.invitro_label,
            )
            test_rf = find_nearest_vitro(
                test_rf,
                args.db_invitro,
                db_invitro_matrix_normed,
                test_index,
                args.invitro_label,
            )
    # print(train_rf.columns)
    df_score = fit_and_predict(
        model,
        train_rf,
        y_train,
        test_rf,
        y_test,
        encoding,
    )
    df_score["neighbors"] = n_neighbors
    df_score["ah"] = ah
    df_score["ap"] = ap
    df_score["fold"] = num
    return df_score


def cal_normalized_matrix(X_traintest, X_valid, ah, ap):

    minmax = MinMaxScaler().fit(X_traintest[non_categorical])

    X_traintest[non_categorical] = minmax.transform(
        X_traintest.loc[:, non_categorical]
    )
    X_valid[non_categorical] = minmax.transform(X_valid.loc[:, non_categorical])

    matrix_valid = dist_matrix(
        X_valid,
        X_traintest,
        non_categorical,
        categorical,
        ah,
        ap,
    )
    matriX_traintest = dist_matrix(
        X_traintest,
        X_traintest,
        non_categorical,
        categorical,
        ah,
        ap,
    )
    return matriX_traintest, matrix_valid



if __name__ == "__main__":

    categorical, conc_column = get_col_ls(args.invitro_type)

 
    if args.encoding == "binary":
        encoding = "binary"
        encoding_value = 1
    elif args.encoding == "multiclass":
        encoding = "multiclass"
        encoding_value = [0.1, 1, 10, 100]

    rand = random.randrange(1, 100)


    # loading data & splitting into train and test dataset

    print("loading dataset...", ctime())
    if args.db_invitro == "no" or args.db_invitro == "overlap":
        db_mortality = load_data(
            args.inputFile,
            encoding=encoding,
            categorical_columns=categorical,
            conc_column=conc_column,
            encoding_value=encoding_value,
            seed=rand,
        )
        db_invitro = args.db_invitro
    else:
        db_mortality, db_invitro = load_invivo_invitro(
            args.inputFile,
            args.inputFile_vitro,
            encoding=encoding,
            encoding_value=encoding_value,
            seed=42,
        )

    print("finish loaded.", ctime())


    db_mortality["fish"] = (
        str(db_mortality["class"])
        + " "
        + str(db_mortality["tax_order"])
        + " "
        + str(db_mortality["family"])
        + " "
        + str(db_mortality["genus"])
        + " "
        + str(db_mortality["species"])
    )
    df_fishchem = db_mortality[["fish", "test_cas", conc_column]]

    idx_record = []
    # saving information for each splitting time 
    for repeat in range(args.repeat):



        logging.basicConfig(
            filename=args.outputFile + "_log_{}.log".format(repeat),
            level=logging.DEBUG,
            format="%(asctime)s:%(levelname)s:%(name)s:%(message)s",
        )
        logging.info("The" + str(repeat) + "th times:")

        grid_search = pd.DataFrame()

        # -----------------startified split the dataset for training model--------
        test_size = 0.2
        col_groups = "test_cas"

        traintest_idx, valid_idx = get_grouped_train_test_split( df_fishchem, test_size, col_groups, rand=repeat)


        if list(valid_idx) in idx_record:
            continue
        else:
            idx_record.append(list(valid_idx))
        df_fishchem_tv = df_fishchem.iloc[traintest_idx, :]

        X = db_mortality.drop(columns=conc_column)


        X_traintest, X_valid, Y_traintest, Y_valid = get_train_test_data(db_mortality, traintest_idx, valid_idx, conc_column)
        

        # -----------------creating the distance matrix--------
        print("calcultaing distance matrix..", ctime())
        matrices = cal_matrixs( X_traintest, X_traintest, categorical, non_categorical)

        matrices_full = cal_matrixs(X, X, categorical, non_categorical)
        
        if args.db_invitro != "no" and args.db_invitro != "overlap":
            matrices_invitro = cal_matrixs(
                X_traintest, db_invitro, categorical_both, non_categorical
            )
        else:
            matrices_invitro = None
        
        print("distance matrix calculation finished", ctime())

        # -----------------hyperparameter range-------------------
        if args.alpha_h == "logspace":
            sequence_ah = np.logspace(-2, 0, 3)
        else:
            sequence_ah = [float(args.hamming_alpha)]

        if args.alpha_p == "logspace":
            sequence_ap = np.logspace(-2, 0, 3)
        else:
            sequence_ap = [float(args.pubchem2d_alpha)]

        # models parameters
        hyper_params_tune,model = set_hyperparameters(args.model_type)

        params_comb = list(
            ParameterSampler(
                hyper_params_tune,
                n_iter=args.niter,
                random_state=rand,
            )
        )

        # -------------------training using the default model setting and found the best alphas combination--------------------
        
        group_kfold = GroupKFold(n_splits=4)

        folds = list(group_kfold.split(df_fishchem_tv, groups=df_fishchem_tv[col_groups]))

        count = 1
        num_runs = (len(sequence_ap) *len(sequence_ah) * len(args.neighbors)  + args.niter )
        best_accs = 0

        for nei in args.neighbors:
            for ah in sequence_ah:
                for ap in sequence_ap:
                    try:
                        results = []

                        print(
                            ah,
                            ap,
                            "*" * 50,
                            count / num_runs,
                            ctime(),
                            end="\r",
                        )
                        count = count + 1

                        with ProcessPool(max_workers=4) as pool:
                            for num, fold in enumerate(folds):
                                y_train = Y_traintest[fold[0]]
                                y_test = Y_traintest[fold[1]]

                                if args.w_invitro == "own":
                                    train_matrix = pd.DataFrame()
                                    test_matrix = pd.DataFrame()
                                else:
                                    (train_matrix,
                                        test_matrix,
                                    ) = get_traintest_matrices(
                                        matrices, fold, ah, ap
                                    )


                                res = pool.schedule(
                                    s_rasar_func,
                                    args=(
                                        fold[0],
                                        fold[1],
                                        train_matrix,
                                        test_matrix,
                                        y_train,
                                        y_test,
                                        X_traintest,
                                        nei,
                                        ah,
                                        ap,
                                        matrices_invitro,
                                        max_euc,
                                        model,
                                    ),
                                    timeout=120,
                                )

                                try:
                                    results.append(res)
                                except TimeoutError as error:
                                    print(
                                        "function took longer than %d seconds"
                                        % error.args[1]
                                    )

                         
                            results = [i.result() for i in results]

                        result_stats = result_describe(results)

                        #---------------testing on the validation dataset---------
 

                        if args.w_invitro == "own":
                            train_rf = pd.DataFrame()
                            test_rf = pd.DataFrame()
                        else:
                            (matrix_traintest, matrix_valid) = get_traintest_matrices(
                                matrices_full, [traintest_idx, valid_idx], ah, ap
                                )

                            train_rf, test_rf = cal_s_rasar(
                                matrix_traintest,
                                matrix_valid,
                                Y_traintest,
                                nei,
                                encoding,
                            )

                        if args.w_invitro != "False":
                            if str(args.db_invitro) == "overlap":
                                train_rf = get_vitroinfo(
                                    train_rf, X, traintest_idx, args.invitro_label
                                )
                                test_rf = get_vitroinfo(
                                    test_rf, X, valid_idx, args.invitro_label
                                )
                            else:
                                db_invitro_matrix_normed = normalized_invitro_matrix(
                                    X_traintest, matrices_invitro, ah, ap
                                )

                                train_rf = find_nearest_vitro(
                                    train_rf,
                                    args.db_invitro,
                                    db_invitro_matrix_normed,
                                    traintest_idx,
                                    args.invitro_label,
                                )
                                test_rf = find_nearest_vitro(
                                    test_rf,
                                    args.db_invitro,
                                    db_invitro_matrix_normed,
                                    valid_idx,
                                    args.invitro_label,
                                )


                        df_test_score = fit_and_predict(
                            model,
                            train_rf,
                            Y_traintest,
                            test_rf,
                            Y_valid,
                            args.encoding,
                        )

                        try:
                            dict_acc = dataset_acc(
                                X_traintest, X_valid, Y_traintest, Y_valid
                            )
                        except:
                            dict_acc = pd.DataFrame()
                        

                        temp_grid_full = save_results(
                            "default",
                            ah,
                            ap,
                            result_stats,
                            df_test_score,
                            dict_acc,
                        )
                        temp_grid_full["num_neighbor"] = int(nei)
                        grid_search = pd.concat([grid_search, temp_grid_full])

                        if result_stats[0].accuracy.values[0] > best_accs:
                            best_accs = result_stats[0].accuracy.values[0]
                            best_ah = ah
                            best_ap = ap
                            print("success found better alphas", best_accs, end="\r")

                    except ValueError as error:
                       logging.error(str(error))



            # --------------explore more on the model's hyperparameter range with found alphas--------


            for j in tqdm(range(0, len(params_comb))):
                print(
                ah,
                ap,
                "*" * 50,
                count / num_runs,
                ctime(),
                end="\r",
                )
                count = count + 1


                for k, v in params_comb[j].items():
                    setattr(model, k, v)
                try:
                    results = []
    
                    with ProcessPool(max_workers=4) as pool:
                        for num, fold in enumerate(folds):
                            y_train = Y_traintest[fold[0]]
                            y_test = Y_traintest[fold[1]]

                            matrix_euc = pd.DataFrame(matrix_euc)
                            max_euc = matrix_euc.iloc[fold[0], fold[0]].values.max()

                            if args.w_invitro == "own":
                                train_matrix = pd.DataFrame()
                                test_matrix = pd.DataFrame()
                            else:
                                (train_matrix,
                                    test_matrix,
                                ) = get_traintest_matrices(
                                    matrices, fold, ah, ap
                                )

                            res = pool.schedule(
                                s_rasar_func,
                                args=(
                                    fold[0],
                                    fold[1],
                                    train_matrix,
                                    test_matrix,
                                    y_train,
                                    y_test,
                                    X_traintest,
                                    nei,
                                    best_ah,
                                    best_ap,
                                    matrices_invitro,
                                    max_euc,
                                    num,
                                    model,
                                ),
                                timeout=120,
                            )

                            try:
                                results.append(res)
                            except TimeoutError as error:
                                print(
                                    "function took longer than %d seconds"
                                    % error.args[1]
                                )
                            del train_matrix, test_matrix

                        results = [i.result() for i in results]


                    result_stats = result_describe(results)


                    #---------------testing on the validation dataset---------

                    if args.w_invitro == "own":
                        train_rf = pd.DataFrame()
                        test_rf = pd.DataFrame()
                    else:
                        (matrix_traintest, matrix_valid) = get_traintest_matrices(
                                matrices_full, [traintest_idx, valid_idx], ah, ap
                                )
                        
                        train_rf, test_rf = cal_s_rasar(
                            matrix_traintest,
                            matrix_valid,
                            Y_traintest,
                            nei,
                            encoding,
                        )

                    if args.w_invitro != "False":
                        if str(args.db_invitro) == "overlap":
                            train_rf = get_vitroinfo(
                                train_rf, X, traintest_idx, args.invitro_label
                            )
                            test_rf = get_vitroinfo(
                                test_rf, X, valid_idx, args.invitro_label
                            )
                        else:
                            db_invitro_matrix_normed = normalized_invitro_matrix(
                                X_traintest, matrices_invitro, best_ah, best_ap
                            )
                            train_rf = find_nearest_vitro(
                                train_rf,
                                args.db_invitro,
                                db_invitro_matrix_normed,
                                traintest_idx,
                                args.invitro_label,
                            )
                            test_rf = find_nearest_vitro(
                                test_rf,
                                args.db_invitro,
                                db_invitro_matrix_normed,
                                valid_idx,
                                args.invitro_label,
                            )


                    df_test_score = fit_and_predict(
                        model,
                        train_rf,
                        Y_traintest,
                        test_rf,
                        Y_valid,
                        args.encoding,
                    )

                    try:
                        dict_acc = dataset_acc(X_traintest, X_valid, Y_traintest, Y_valid)
                    except:
                        dict_acc = pd.DataFrame()

                    temp_grid = save_results(
                        params_comb[j],
                        best_ah,
                        best_ap,
                        result_stats,
                        df_test_score,
                        dict_acc
                    )
                    
                    if result_stats[0].accuracy.values[0] > best_accs:
                        best_accs = result_stats[0].accuracy.values[0]
                        best_score = temp_grid
                        print("success found better hyperparameters", best_accs)

                    grid_search = pd.concat([grid_search, temp_grid])

                except ValueError as error:
                    logging.error(str(error))

        # ----------------save the information into a file-------
        df2file(grid_search, args.outputFile + "_fullinfo_{}.txt".format(repeat))
        df2file(best_score,args.outputFile + "_{}.txt".format(repeat))

# -------------------------------------------------invivo to invivo (gvv)-------------------------------------------------
# binary:
# python RASAR_mulneigh_bi_cte_test_RF.py -i '/local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv' -r 1 -n 1 -ni 5000  -o "general/vivo/s_rasar_rasar_bi_1_v2"


# ------------------------------------------------------invitro to invitro(gtt)----------------------------------------------------------
# binary:
# python RASAR_mulneigh_bi_cte_test_RF.py     -i /local/wujimeng/code_jimeng/data/invitro/invitro_processed.csv  -r 1  -ni 5000  -iv eawag   -n 1  -o general/vitro/vitro_s_rasar_bi_1_v2


# cd /local/wujimeng/code_jimeng/c_cte
# source activate rdkit-test


# ------------------------------------------------------invitro + invivo to invivo(otv)----------------------------------------------------------
# binary & R unrelated
# python RASAR_mulneigh_bi_cte_test_RF.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv  -ah 1  -ap 1 -n 1  -ni 20000   -dbi "overlap" -wi "own"   -il "label"                -o "vitro_e/vivo+vitro/R1/S/test/RF_tuned/try2/repeat_own_label"
# python RASAR_mulneigh_bi_cte_test_RF.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv  -ah 1  -ap 1 -n 1  -ni 20000   -dbi "overlap" -wi "own"   -il "number"               -o "vitro_e/vivo+vitro/R1/S/test/RF_tuned/try2/repeat_own_number"
# python RASAR_mulneigh_bi_cte_test_RF.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv  -ah 1  -ap 1 -n 1  -ni 20000   -dbi "overlap" -wi "own"   -il "label_half"           -o "vitro_e/vivo+vitro/R1/S/test/RF_tuned/try2/repeat_own_label_half"


# ------------------------------------------------------invitro + invivo to invivo(otvv)----------------------------------------------------------
# binary & R=1
# python RASAR_mulneigh_bi_cte_test_LR.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv  -n 1  -ni 20000 -dbi "overlap" -wi "True"   -il "label"                -o "vitro_e/vivo+vitro/R1/S/test/LR/repeat_label"

# python RASAR_mulneigh_bi_cte_test_RF.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv  -n 1  -ni 20000 -ah 0.143845 -ap 0.069519  -dbi "overlap" -wi "True"   -il "label"                -o "vitro_e/vivo+vitro/R1/S/test/RF_tuned/try2/repeat_label"
# python RASAR_mulneigh_bi_cte_test_RF.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv  -n 1  -ni 20000 -ah 0.143845 -ap 0.069519 -dbi "overlap" -wi "True"   -il "number"               -o "vitro_e/vivo+vitro/R1/S/test/RF_tuned/try2/repeat_number"
# python RASAR_mulneigh_bi_cte_test_RF.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv  -n 1  -ni 20000 -ah 0.143845 -ap 0.069519 -dbi "overlap" -wi "True"   -il "label_half"           -o "vitro_e/vivo+vitro/R1/S/test/RF_tuned/try2/repeat_label_half"

# binary & R=4
# python RASAR_mulneigh_bi_cte_test_RF.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv  -n 4     -dbi "overlap" -wi "True"   -il "label"                -o "vitro_e/vivo+vitro/R4/S/test/RF/repeat_label"
# python RASAR_mulneigh_bi_cte_test_RF.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv  -n 4     -dbi "overlap" -wi "True"   -il "number"               -o "vitro_e/vivo+vitro/R4/S/test/RF/repeat_number"
# python RASAR_mulneigh_bi_cte_test_RF.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv  -n 4     -dbi "overlap" -wi "True"   -il "label_half"           -o "vitro_e/vivo+vitro/R4/S/test/RF/repeat_label_half"


# ------------------------------------------------------invivo to sinvivo(ovv)----------------------------------------------------------
# binary & R=1
# python RASAR_mulneigh_bi_cte_test_RF.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv -ah 0.143845 -ap 0.069519 -n 1 -ni 30000  -o "vitro_e/vivo+vitro/R1/S/test/RF_tuned/try2/repeat"

# binary & R=4
# python RASAR_mulneigh_bi_cte_test_RF.py -i /local/wujimeng/code_jimeng/data/invivo/invivo_repeated_w_invitro_eawag.csv  -n 4 -ni 20000  -o "vitro_e/vivo+vitro/R4/S/test/RF/repeat"


# def save_results(
#     j,
#     ah,
#     ap,
#     df_mean,
#     df_std,
#     df_test_score,
#     X_traintest,
#     X_valid,
#     Y_traintest,
#     Y_valid,
# ):
#     temp_grid = pd.DataFrame()
#     if j != None:
#         for k, v in params_comb[j].items():
#             temp_grid = pd.concat([temp_grid, pd.DataFrame([v], columns=[k])], axis=1)
#     temp_grid = pd.concat([temp_grid, pd.DataFrame([ah], columns=["ah"])], axis=1)
#     temp_grid = pd.concat([temp_grid, pd.DataFrame([ap], columns=["ap"])], axis=1)
#     temp_grid = pd.concat(
#         [
#             temp_grid,
#             df_mean.add_prefix("train_avg_"),
#             df_std.add_prefix("train_std_"),
#             df_test_score.add_prefix("test_"),
#         ],
#         axis=1,
#     )
#     try:
#         test_total = X_valid.shape[0]
#         test_correct = np.sum(X_valid.invitro_label.values == Y_valid)
#         test_acc = test_correct / test_total
#         train_total = X_traintest.shape[0]
#         train_correct = np.sum(X_traintest.invitro_label.values == Y_traintest)
#         train_acc = train_correct / train_total
#         temp_grid["train_dataset_accuracy"] = (
#             str(round(train_acc, 4))
#             + " ("
#             + str(train_total)
#             + "/"
#             + str(train_correct)
#             + ")"
#         )

#         temp_grid["test_dataset_accuracy"] = (
#             str(round(test_acc, 4))
#             + " ("
#             + str(test_total)
#             + "/"
#             + str(test_correct)
#             + ")"
#         )

#     except:
#         pass

#     return temp_grid
