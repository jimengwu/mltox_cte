import numpy as np
from numpy.lib.stride_tricks import as_strided
import pandas as pd
import random
import datetime
import os
import logging
import warnings
import heapq
from collections import Counter
from math import sqrt
from tqdm import tqdm
from time import ctime

import h2o
from h2o.estimators.random_forest import H2ORandomForestEstimator

from scipy.stats import sem
from scipy.spatial.distance import pdist, squareform, cdist

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, DistanceMetric
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, LabelBinarizer
from sklearn.metrics import (
    recall_score,
    confusion_matrix,
    precision_score,
    accuracy_score,
    mean_squared_error,
    f1_score,
)
from sklearn.model_selection import (
    KFold,
    GroupKFold,
    train_test_split,
    GroupShuffleSplit,
    ParameterSampler,
)


warnings.filterwarnings("ignore")

# categorical features will ordinal encoded

categorical = [
    "class",
    "tax_order",
    "family",
    "genus",
    "species",
    "control_type",
    "media_type",
    "application_freq_unit",
    "exposure_type",
    "conc1_type",
    "obs_duration_mean",
]

categorical_invitro = [
    "cell_line",
    "endpoint",
    "timepoint_hour",
    "insert",
    "passive_dosing",
    "plate_size",
    "solvent",
    "conc_determination_nominal_or_measured",
    "medium",
    "fbs",
]

categorical_both = [
    "class",
    "tax_order",
    "family",
    "genus",
    "species",
]  # for in vitro and in vivo


# non_categorical was numerical features, whcih will be standarized. \
# Mol,bonds_number, atom_number was previously log transformed due to the maginitude of their values.

non_categorical = [
    "ring_number",
    "tripleBond",
    "doubleBond",
    "alone_atom_number",
    "oh_count",
    "atom_number",
    "bonds_number",
    "MorganDensity",
    "LogP",
    "mol_weight",
    "water_solubility",
    "melting_point",
]
# comparing was used to identify similar experiments
comparing = ["test_cas"] + categorical

non_categorical_invitro = [
    "logkow",
    "water_solubility",
    "henry_constant",
    "atom_number",
    "alone_atom_number",
    "bonds_number",
    "doubleBond",
    "tripleBond",
    "ring_number",
    "mol_weight",
    "MorganDensity",
    "LogP",
    "oh_count",
    "melting_point",
    "ntc",
]


tqdm.pandas(desc="Bar")


# ------------------------------------------------------------ input  preprocess: data labelling and encoding, model parameter setting----------


def encoding_categorical_invitro(cat_cols, database_1, database_2, database_3):
    # fit to the categories in datafusion dataset and training dataset
    categories = []
    for column in database_1[cat_cols]:
        unique_db_3 = tuple(
            i
            for i in database_3[column].unique()
            if i not in database_1[column].unique()
        )
        unique_db_2 = tuple(
            i
            for i in database_2[column].unique()
            if i not in database_1[column].unique()
        )

        cat_final = np.concatenate(
            (database_1[column].unique(), unique_db_2, unique_db_3), axis=0
        )

        cat_final = np.array(sorted(cat_final))
        categories.append(cat_final)

    encoder = OrdinalEncoder(categories=categories, dtype=int)

    encoder.fit(database_1[cat_cols])
    database_1[cat_cols] = encoder.transform(database_1[cat_cols]) + 1

    encoder.fit(database_2[cat_cols])
    database_2[cat_cols] = encoder.transform(database_2[cat_cols]) + 1

    encoder.fit(database_3[cat_cols])
    database_3[cat_cols] = encoder.transform(database_3[cat_cols]) + 1

    return database_1, database_2, database_3


def multiclass_encoding(var, threshold=[10 ** -1, 10 ** 0, 10 ** 1, 10 ** 2]):
    var_ls = list(var)
    for i in range(len(var_ls)):
        if var_ls[i] <= threshold[0]:
            var_ls[i] = 4

        elif threshold[0] < var_ls[i] <= threshold[1]:
            var_ls[i] = 3

        elif threshold[1] < var_ls[i] <= threshold[2]:
            var_ls[i] = 2

        elif threshold[2] < var_ls[i] <= threshold[3]:
            var_ls[i] = 1

        else:
            var_ls[i] = 0
    return pd.to_numeric(var_ls, downcast="integer")


def encoding_categorical(cat_cols, database, database_df):

    # fit to the categories in datafusion dataset and training dataset
    categories = []
    for column in database[cat_cols]:
        # add non-existing new category in df dataset but not in training dataset.
        cat_final = np.append(
            database[column].unique(),
            tuple(
                i
                for i in database_df[column].unique()
                if i not in database[column].unique()
            ),
        )
        cat_final = np.array(sorted(cat_final))
        categories.append(cat_final)

    encoder = OrdinalEncoder(categories=categories, dtype=int)

    encoder.fit(database[cat_cols])
    database[cat_cols] = encoder.transform(database[cat_cols]) + 1

    encoder.fit(database_df[cat_cols])
    database_df[cat_cols] = encoder.transform(database_df[cat_cols]) + 1

    return database, database_df


def get_col_ls(label):
    """
    This function returns categorical feature list and concentration column name given the input file type, \
        if the in vitro file comes from Utox lab in eawag Toxcast database, both, or input data is not in vitro dataset.
    """
    # categorical features will ordinal encoded
    # non_categorical was numerical features, whcih will be standarized. \
    # Mol,bonds_number, atom_number was previously log transformed due to the maginitude of their values.

    try:
        if label == "both":
            categorical_ls = ["class", "tax_order", "family", "genus", "species"]
            conc_column = "conc1_mean"
        elif label == "eawag":
            categorical_ls = [
                "class",
                "tax_order",
                "family",
                "genus",
                "species",
                "cell_line",
                "endpoint",
                "solvent",
                "conc_determination_nominal_or_measured",
            ]
            conc_column = "ec50"
        elif label == "toxcast":
            categorical_ls = [
                "class",
                "tax_order",
                "family",
                "genus",
                "species",
                # "modl"
            ]
            conc_column = "conc"
        else:
            conc_column = "conc1_mean"
            categorical_ls = categorical
        return categorical_ls, conc_column
    except:
        pass


def change_label(x):
    return x.str.replace("", " ").str.strip().str.split(" ")


def set_hyperparameters(model_type):
    if model_type == "LR":
        model = LogisticRegression(random_state=1)

        hyper_params_tune = {
            "penalty": ["l1", "l2", "elasticnet", "none"],
            "dual": [True, False],
            "fit_intercept": [True, False],
            # "intercept_scaling": np.logspace(-4, 4, 10),
            "intercept_scaling": np.logspace(-4, 4, 10),
            "C": np.logspace(-4, 4, 10),
            "criterion": ["gini", "entropy", "log_loss"],
            "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
            # "max_iter": np.logspace(0, 3, 10),
            "max_iter": np.logspace(0, 3, 10),
            "multi_class": ["auto", "ovr", "multinomial"],
            "warm_start": [True, False],
            "l1_ratio": np.logspace(-1, 0, 10),
            "class_weight": [{0: i, 1: 1} for i in range(0, 10)],
            "tol": np.logspace(-3, 3, 8),
        }

    elif model_type == "RF":
        model = RandomForestClassifier(random_state=1)

        hyper_params_tune = {
            # "n_estimators": [int(x) for x in np.linspace(start=1, stop=5, num=5)],
            # "max_features": [i for i in range(1, 4, 1)],
            # "max_depth": [i for i in np.linspace(start=0.01, stop=1, num=5)],
            # "min_samples_leaf": [i for i in np.linspace(start=0.01, stop=0.5, num=5)],
            # "min_samples_split": [i for i in np.linspace(start=0.1, stop=1, num=5)],
            # ---------------------------------
            # "n_estimators": [int(x) for x in np.linspace(start=1, stop=150, num=5)],
            # # "max_features": [i for i in range(1, 4, 1)],
            # "max_depth": [i for i in range(1, 10, 2)]
            # + [i for i in range(10, 100, 20)]
            # + [None],
            # "min_samples_leaf": [i for i in np.linspace(start=0.1, stop=0.5, num=5)],
            # "min_samples_split": [i for i in np.linspace(start=0.1, stop=1, num=5)],
            # "max_samples": [i for i in np.linspace(start=0.1, stop=1, num=5)],
            # --------------------------------------------
            "n_estimators": [int(x) for x in np.linspace(start=10, stop=100, num=5)]
            + [int(x) for x in np.linspace(start=2, stop=5, num=4)],
            "max_depth": [i for i in np.linspace(start=0.01, stop=1, num=5)]
            + [i for i in range(2, 10, 1)]
            + [i for i in range(10, 100, 20)]
            + [None],
            "min_samples_leaf": [i for i in np.linspace(start=0.01, stop=0.5, num=5)]
            + [i for i in np.linspace(start=0.1, stop=0.4, num=4)]
            + [1, 2, 4],
            "min_samples_split": [i for i in np.linspace(start=0.1, stop=1, num=5)]
            + [2, 4, 6],
            "max_samples": [i for i in np.linspace(start=0.1, stop=0.9, num=5)]
            + [i for i in np.linspace(start=0.2, stop=0.9, num=8)],
            # ---------------------------------------------
            # "n_estimators": [int(x) for x in np.linspace(start=1, stop=150, num=5)],
            # "max_depth": [i for i in range(1, 10, 2)]
            # + [i for i in range(10, 100, 20)]
            # + [None],
            # "min_samples_leaf": [i for i in np.linspace(start=0.1, stop=0.5, num=5)],
            # "min_samples_split": [i for i in np.linspace(start=0.1, stop=1, num=5)],
            # "max_samples": [i for i in np.linspace(start=0.1, stop=1, num=5)],
        }

    elif model_type == "DT":
        hyper_params_tune = {
            "criterion": ["gini", "entropy"],
            "splitter": ["best", "random"],
            "max_depth": [int(x) for x in np.linspace(start=10, stop=100, num=5)],
            "min_samples_split": [
                int(x) for x in np.linspace(start=10, stop=100, num=5)
            ],
            "min_samples_leaf": [
                int(x) for x in np.linspace(start=10, stop=100, num=5)
            ],
            "min_weight_fraction_leaf": [
                x for x in np.linspace(start=0, stop=0.5, num=5)
            ],
            "max_features": ["auto", "sqrt", "log2"],
            "max_leaf_nodes": [int(x) for x in np.linspace(start=10, stop=100, num=5)],
            "min_impurity_decrease": [
                i for i in np.linspace(start=0.01, stop=1, num=5)
            ],
            "class_weight": [{0: i, 1: 1} for i in range(0, 10)],
            "ccp_alphanon-negative": [
                i for i in np.linspace(start=0.01, stop=1, num=5)
            ],
        }
        model = DecisionTreeClassifier(random_state=1)

    return hyper_params_tune, model


def vitroconc_to_label(df, thres):
    df["invitro_label"] = np.where(df["invitro_conc"] > thres, 0, 1)
    return df


def add_fish_column(df):
    df["fish"] = (
        str(df["class"])
        + " "
        + str(df["tax_order"])
        + " "
        + str(df["family"])
        + " "
        + str(df["genus"])
        + " "
        + str(df["species"])
    )
    return df


# --------------------------------------------------------------data splitting------------------
def get_train_test_data(db_mortality, traintest_idx, valid_idx, conc_column):

    X_trainvalid = db_mortality.drop(columns=conc_column).iloc[traintest_idx, :]
    X_valid = db_mortality.drop(columns=conc_column).iloc[valid_idx, :]
    Y_trainvalid = db_mortality.iloc[traintest_idx, :][conc_column].values
    Y_valid = db_mortality.iloc[valid_idx, :][conc_column].values
    return X_trainvalid, X_valid, Y_trainvalid, Y_valid


def get_grouped_train_test_split(df_all, test_size, col_groups, rand=2):
    """
    get a grouped train test split that is within +- 0.05 of the specified test_size

    """
    test_size_actual = 1.0
    print(df_all.shape[0])
    random.seed(rand)
    while (test_size_actual < test_size - 0.05) | (test_size_actual > test_size + 0.05):

        trainvalid_idx, test_idx = next(
            GroupShuffleSplit(
                test_size=test_size, n_splits=2, random_state=random.randint(1, 100)
            ).split(df_all, groups=df_all[col_groups])
        )
        test_size_actual = len(test_idx) / len(df_all)

        print(test_size_actual)

    return trainvalid_idx, test_idx


# -----------------------------------------------------------------results-------------------


def fit_and_predict(model, X_train, y_train, X_test, y_test, encoding="binary"):
    """fit the model and predict the score on the test data."""
    df_output = pd.DataFrame()

    # print("start fitting", end="\r")
    model.fit(X_train, y_train)
    # print("finish fitting", end="\r")
    y_pred = model.predict(X_test)
    # print("predict finish", end="\r")
    if encoding == "binary":

        df_output.loc[0, "accuracy"] = accuracy_score(y_test, y_pred)
        df_output.loc[0, "recall"] = recall_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        df_output.loc[0, "specificity"] = tn / (tn + fp)
        df_output.loc[0, "f1"] = f1_score(y_test, y_pred)
        df_output.loc[0, "precision"] = precision_score(y_test, y_pred)

    elif encoding == "multiclass":
        df_output.loc[0, "accuracy"] = accuracy_score(y_test, y_pred)
        df_output.loc[0, "recall"] = recall_score(y_test, y_pred, average="macro")
        df_output.loc[0, "specificity"] = np.nan
        df_output.loc[0, "f1"] = f1_score(y_test, y_pred, average="macro")
        df_output.loc[0, "precision"] = precision_score(y_test, y_pred, average="macro")
    return df_output


def dataset_acc(trainvalid, valid, label_trainvalid, label_valid):
    """compare the invitro label and predicted invivo label"""
    dict_acc = {}

    dict_acc["test_total"] = valid.shape[0]
    dict_acc["test_correct"] = np.sum(valid.invitro_label.values == label_valid)
    dict_acc["test_acc"] = dict_acc["test_correct"] / dict_acc["test_total"]
    dict_acc["train_total"] = trainvalid.shape[0]
    dict_acc["train_correct"] = np.sum(
        trainvalid.invitro_label.values == label_trainvalid
    )
    dict_acc["train_acc"] = dict_acc["train_correct"] / dict_acc["train_total"]

    return dict_acc


def result_describe(results):
    df_output = pd.concat(results, axis=0)
    df_mean = pd.DataFrame(df_output.mean(axis=0)).transpose()
    df_std = pd.DataFrame(df_output.sem(axis=0)).transpose()
    return df_mean, df_std


def save_results(params, ah, ap, result_stats, df_test_score, dict_acc,thres="median"):

    temp_grid = pd.DataFrame()
    if params == "default":
        temp_grid = pd.concat(
            [temp_grid, pd.DataFrame([params], columns=["params"])], axis=1,
        )
    else:
        for k, v in params.items():
            temp_grid = pd.concat([temp_grid, pd.DataFrame([v], columns=[k])], axis=1,)

    temp_grid = pd.concat(
        [
            temp_grid,
            result_stats[0].add_prefix("train_avg_"),
            result_stats[1].add_prefix("train_std_"),
            df_test_score.add_prefix("test_"),
        ],
        axis=1,
    )
    temp_grid["threshold"] = thres
    temp_grid["alpha_h"] = ah
    temp_grid["alpha_p"] = ap
    try:
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
    except:
        pass

    return temp_grid


def df2file(info, outputFile):
    """save the dataframe into file."""
    filename = outputFile
    dirname = os.path.dirname(filename)
    if (not os.path.exists(dirname)) & (dirname != ""):
        os.makedirs(dirname)
    info.to_csv(filename)
    print("\n", "Result saved.")


# -------------------------------------------------------------------load data-------------------------------
def load_data_mlp(DATA_PATH, encoding, encoding_value=1, seed=42):
    db = pd.read_csv(DATA_PATH).drop(columns=["Unnamed: 0"]).drop_duplicates()
    db_raw = db.copy()

    # MinMax trasform for numerical variables
    for nc in non_categorical:
        minmax = MinMaxScaler()
        minmax.fit(db[[nc]])
        db[[nc]] = minmax.transform(db[[nc]])

    # One hot Encoding for categorical variables
    for i in categorical:
        jobs_encoder = LabelBinarizer()
        jobs_encoder.fit(db[i])
        transformed = jobs_encoder.transform(db[i])
        if jobs_encoder.classes_.shape[0] > 2:
            ohe_df = pd.DataFrame(
                transformed,
                columns=[
                    str(i) + "_" + str(j) for j in range(jobs_encoder.classes_.shape[0])
                ],
            )
        else:
            ohe_df = pd.DataFrame(transformed, columns=[str(i) + "_0"])
        db = pd.concat([db, ohe_df], axis=1).drop([i], axis=1)

    # # Encoding for target variable: binary and multiclass
    if encoding == "binary":
        db["conc1_mean"] = np.where(db["conc1_mean"].values > encoding_value, 0, 1)
        print(encoding_value)

    elif encoding == "multiclass":
        label = db["conc1_mean"].copy()
        db["conc1_mean"] = multiclass_encoding(label, encoding_value)
        print(encoding_value)

    X = db.drop(columns="conc1_mean")
    Y = db["conc1_mean"].values
    return X, Y


def load_data_mlp_repeat(DATA_PATH1, DATA_PATH2, encoding, encoding_value=1, seed=42):
    db1 = pd.read_csv(DATA_PATH1).drop(columns=["Unnamed: 0"]).drop_duplicates()
    db2 = pd.read_csv(DATA_PATH2).drop(columns=["Unnamed: 0"]).drop_duplicates()
    db = pd.concat([db1, db2], axis=0).reset_index(drop=True)
    # MinMax trasform for numerical variables
    for nc in non_categorical:
        minmax = MinMaxScaler()
        minmax.fit(db[[nc]])
        db[[nc]] = minmax.transform(db[[nc]])

    # One hot Encoding for categorical variables
    for i in categorical:
        jobs_encoder = LabelBinarizer()
        jobs_encoder.fit(db[i])
        transformed = jobs_encoder.transform(db[i])
        if jobs_encoder.classes_.shape[0] > 2:
            ohe_df = pd.DataFrame(
                transformed,
                columns=[
                    str(i) + "_" + str(j) for j in range(jobs_encoder.classes_.shape[0])
                ],
            )
        else:
            ohe_df = pd.DataFrame(transformed)
        db = pd.concat([db, ohe_df], axis=1).drop([i], axis=1)

    # Encoding for target variable: binary and multiclass
    if encoding == "binary":
        db["conc1_mean"] = np.where(db["conc1_mean"].values > encoding_value, 0, 1)
        print(encoding_value)

    elif encoding == "multiclass":
        label = db["conc1_mean"].copy()
        db["conc1_mean"] = multiclass_encoding(label, encoding_value)
        print(encoding_value)

    X_norepeat = db[db1.shape[0] :].reset_index(drop=True).drop(columns="conc1_mean")
    Y_norepeat = db[db1.shape[0] :].reset_index(drop=True)["conc1_mean"].values
    X_repeat = db[: db1.shape[0]].reset_index(drop=True).drop(columns="conc1_mean")
    Y_repeat = db[: db1.shape[0]].reset_index(drop=True)["conc1_mean"].values
    return X_repeat, Y_repeat, X_norepeat, Y_norepeat


def load_data(
    DATA_PATH,
    encoding,
    categorical_columns,
    drop_columns=["Unnamed: 0"],
    conc_column="conc1_mean",
    encoding_value=1,
    seed=42,
):
    db = pd.read_csv(DATA_PATH).drop(drop_columns, axis=1)

    db_pub = (
        db["pubchem2d"].str.replace("", " ").str.strip().str.split(" ", expand=True)
    )
    db_pub.columns = ["pub" + str(i) for i in range(1, 882)]
    db_pub["test_cas"] = db["test_cas"]
    db_pub.drop_duplicates(inplace=True)
    db = db.merge(db_pub, on="test_cas").reset_index(drop=True)
    db.drop(columns="pubchem2d", inplace=True)

    # Ordinal Encoding for categorical variables
    encoder = OrdinalEncoder(dtype=int)
    encoder.fit(db[categorical_columns])
    db[categorical_columns] = encoder.transform(db[categorical_columns]) + 1

    # Encoding for target variable: binary and multiclass
    if encoding == "binary":
        if encoding_value == "own":
            print(
                "labelling the experiment as "
                + encoding
                + " using "
                + str(np.median(db[conc_column]))
            )
            db[conc_column] = np.where(
                db[conc_column].values > np.median(db[conc_column]), 0, 1
            )
        else:
            print(
                "labelling the experiment as "
                + encoding
                + " using "
                + str(encoding_value)
            )
            db[conc_column] = np.where(db[conc_column].values > encoding_value, 0, 1)

    elif encoding == "multiclass":
        if encoding_value == "own":
            print(
                "labelling the experiment as "
                + encoding
                + " using "
                + str(db[conc_column].quantile([0.2, 0.4, 0.6, 0.8]).values)
            )
            db[conc_column] = multiclass_encoding(
                db[conc_column].copy(),
                db[conc_column].quantile([0.2, 0.4, 0.6, 0.8]).values,
            )
        else:
            print(
                "labelling the experiment as "
                + encoding
                + " using "
                + str(encoding_value)
            )
            db[conc_column] = multiclass_encoding(
                db[conc_column].copy(), encoding_value
            )
    db = add_fish_column(db)
    return db


def load_data_fish(DATA_PATH, DATA_PATH_fish, encoding, encoding_value=1, seed=42):
    db = pd.read_csv(DATA_PATH).drop(columns=["Unnamed: 0"]).drop_duplicates()
    db_fish = pd.read_csv(DATA_PATH_fish).drop(columns=["Unnamed: 0"]).drop_duplicates()
    db_raw = db_fish.copy()

    # MinMax trasform for numerical variables
    for nc in non_categorical:
        minmax = MinMaxScaler()
        minmax.fit(db[[nc]])
        db[[nc]] = minmax.transform(db[[nc]])
        db_fish[[nc]] = minmax.transform(db_fish[[nc]])

    # Ordinal Encoding for categorical variables
    db, db_fish = encoding_categorical(categorical, db, db_fish)

    # Encoding for target variable: binary and multiclass
    if encoding == "binary":
        db["conc1_mean"] = np.where(db["conc1_mean"].values > encoding_value, 0, 1)
        db_fish["conc1_mean"] = np.where(
            db_fish["conc1_mean"].values > encoding_value, 0, 1
        )
        print(encoding_value)

    elif encoding == "multiclass":
        db["conc1_mean"] = multiclass_encoding(db["conc1_mean"].copy(), encoding_value)
        db_fish["conc1_mean"] = multiclass_encoding(
            db_fish["conc1_mean"].copy(), encoding_value
        )
        print(encoding_value)

    X = db.drop(columns="conc1_mean")
    Y = db["conc1_mean"].values
    X_fish = db_fish.drop(columns="conc1_mean")
    Y_fish = db_fish["conc1_mean"].values
    return X, Y, X_fish, Y_fish


def load_invivo_invitro(
    DATA_PATH, DATA_PATH_invitro, encoding, encoding_value=1, seed=42
):
    # load invivo & invitro datasets
    db = pd.read_csv(DATA_PATH).drop(columns=["Unnamed: 0"]).drop_duplicates()
    db_invitro = (
        pd.read_csv(DATA_PATH_invitro).drop(columns=["Unnamed: 0"]).drop_duplicates()
    )
    # db_invitro = db_invitro[non_categorical + ["test_cas", "invitro_conc", "pubchem2d"]]

    # db_invitro["species"] = "mykiss"
    # db_invitro["class"] = "Actinopterygii"
    # db_invitro["tax_order"] = "Salmoniformes"
    # db_invitro["family"] = "Salmonidae"
    # db_invitro["genus"] = "Oncorhynchus"

    # MinMax trasform for numerical variables
    for nc in non_categorical:
        minmax = MinMaxScaler()
        minmax.fit(db[[nc]])
        db[[nc]] = minmax.transform(db[[nc]])
        db_invitro[[nc]] = minmax.transform(db_invitro[[nc]])

    # Ordinal Encoding for categorical variables
    db, db_invitro = encoding_categorical(categorical_both, db, db_invitro)

    cate = [i for i in categorical if i not in categorical_both]
    encoder = OrdinalEncoder(dtype=int)
    encoder.fit(db[cate])
    db[cate] = encoder.transform(db[cate]) + 1

    try:
        db_invitro = db_invitro.rename(columns={"ec50": "invitro_conc"})
    except:
        pass
    try:
        db_invitro = db_invitro.rename(columns={"conc": "invitro_conc"})
    except:
        pass

    # Encoding for target variable: binary and multiclass
    if encoding == "binary":
        db["conc1_mean"] = np.where(db["conc1_mean"].values > encoding_value, 0, 1)
        print(encoding_value)

    elif encoding == "multiclass":
        db["conc1_mean"] = multiclass_encoding(db["conc1_mean"].copy(), encoding_value)
        print(encoding_value)

    # X = db.drop(columns="conc1_mean")
    # Y = db["conc1_mean"].values

    return db, db_invitro


def load_datafusion_datasets(
    DATA_MORTALITY_PATH,
    DATA_OTHER_ENDPOINT_PATH,
    categorical_columns,
    drop_columns=["Unnamed: 0"],
    conc_column="conc1_mean",
    fixed="no",
    encoding="binary",
    encoding_value=1,
    seed=42,
):

    db_datafusion = pd.read_csv(DATA_OTHER_ENDPOINT_PATH).drop(columns=drop_columns)
    db_mortality = pd.read_csv(DATA_MORTALITY_PATH).drop(columns=drop_columns)

    db_pub = pd.concat(
        [
            db_mortality["test_cas"],
            pd.DataFrame(
                pd.DataFrame(db_mortality["pubchem2d"].values)
                .apply(
                    lambda x: x.str.replace("", " ").str.strip().str.split(" "), axis=1
                )[0]
                .to_list(),
                columns=["pub" + str(i) for i in range(1, 882)],
            ),
        ],
        axis=1,
    ).drop_duplicates()
    db_pub_df = pd.concat(
        [
            db_datafusion["test_cas"],
            pd.DataFrame(
                pd.DataFrame(db_datafusion["pubchem2d"].values)
                .apply(
                    lambda x: x.str.replace("", " ").str.strip().str.split(" "), axis=1
                )[0]
                .to_list(),
                columns=["pub" + str(i) for i in range(1, 882)],
            ),
        ],
        axis=1,
    ).drop_duplicates()

    db_mortality = db_mortality.merge(db_pub, on="test_cas").reset_index(drop=True)
    db_mortality.drop(columns="pubchem2d", inplace=True)

    db_datafusion = db_datafusion.merge(db_pub_df, on="test_cas").reset_index(drop=True)
    db_datafusion.drop(columns="pubchem2d", inplace=True)

    db_raw = db_mortality.copy()

    # Ordinal encoding for categorical variables
    db_mortality, db_datafusion = encoding_categorical(
        categorical_columns, db_mortality, db_datafusion
    )
    #  MinMax transform for numerical variables--other effects & endpoints
    for nc in non_categorical:
        minmax = MinMaxScaler()
        minmax.fit(db_datafusion[[nc]])
        db_datafusion[[nc]] = minmax.transform(db_datafusion[[nc]])

    # concentration class labeling
    if encoding == "binary":
        db_mortality[conc_column] = np.where(
            db_mortality[conc_column].values > encoding_value, 0, 1
        )

        for ef in db_datafusion.effect.unique():
            conc = db_datafusion.loc[db_datafusion.effect == ef, conc_column]
            if fixed == "yes":
                class_threshold = pd.read_csv("data/threshold.csv")
                # print("fixed threshold")
                threshold = class_threshold[class_threshold.effect == ef]
                db_datafusion.loc[db_datafusion.effect == ef, "conc1_mean"] = np.where(
                    conc > float(threshold.ths_binary), 0, 1
                )
            else:
                # print('unfixed threshold')
                db_datafusion.loc[db_datafusion.effect == ef, "conc1_mean"] = np.where(
                    conc > np.median(conc), 0, 1
                )

        # mortality always classified using 1
        # db_datafusion.loc[db_datafusion.effect == 'MOR', 'conc1_mean'] = np.where(conc > encoding_value, 0, 1)
    elif encoding == "multiclass":

        t = db_mortality["conc1_mean"].copy()
        db_mortality["conc1_mean"] = multiclass_encoding(t, encoding_value)
        for ef in db_datafusion.effect.unique():
            conc = db_datafusion.loc[db_datafusion.effect == ef, "conc1_mean"].copy()
            if fixed == "yes":
                class_threshold = pd.read_csv("data/threshold.csv")
                print("fixed threshold")
                threshold = class_threshold[class_threshold.effect == ef]
                db_datafusion.loc[
                    db_datafusion.effect == ef, "conc1_mean"
                ] = multiclass_encoding(
                    conc.values,
                    [
                        float(threshold.ths_1),
                        float(threshold.ths_2),
                        float(threshold.ths_3),
                        float(threshold.ths_4),
                    ],
                )
            else:
                # print("unfixed threshold")
                db_datafusion.loc[
                    db_datafusion.effect == ef, "conc1_mean"
                ] = multiclass_encoding(
                    conc.values, conc.quantile([0.2, 0.4, 0.6, 0.8]).values
                )
                # db_datafusion.loc[db_datafusion.effect == ef,
                #                   "conc1_mean"] = multiclass_encoding(
                #                       conc.values, [0.1, 1, 10, 100])
        # mortality always classified using 0.1，1，10，100
        # conc = db_datafusion.loc[db_datafusion.effect == 'MOR', 'conc1_mean'].copy()
        # db_datafusion.loc[db_datafusion.effect == 'MOR', 'conc1_mean'] = multiclass_encoding(conc.values, encoding_value)
    return db_mortality, db_datafusion


def load_datafusion_datasets_invitro(
    DATA_MORTALITY_PATH,
    DATA_OTHER_ENDPOINT_PATH,
    DATA_INVITRO_PATH,
    categorical_columns,
    drop_columns=["Unnamed: 0"],
    conc_column="conc1_mean",
    fixed="no",
    encoding="binary",
    encoding_value=1,
    seed=42,
):

    # class_threshold = pd.read_csv("data/threshold.csv")
    db_datafusion = (
        pd.read_csv(DATA_OTHER_ENDPOINT_PATH)
        .drop(columns=drop_columns)
        .drop_duplicates()
    )
    db_mortality = (
        pd.read_csv(DATA_MORTALITY_PATH).drop(columns=drop_columns).drop_duplicates()
    )
    db_invitro = (
        pd.read_csv(DATA_INVITRO_PATH).drop(columns=["Unnamed: 0"]).drop_duplicates()
    )

    # db_invitro = db_invitro[non_categorical + categorical_both + ["test_cas", "invitro_conc", "pubchem2d"]]

    db_pub = pd.concat(
        [
            db_mortality["test_cas"],
            pd.DataFrame(
                pd.DataFrame(db_mortality["pubchem2d"].values)
                .apply(
                    lambda x: x.str.replace("", " ").str.strip().str.split(" "), axis=1
                )[0]
                .to_list(),
                columns=["pub" + str(i) for i in range(1, 882)],
            ),
        ],
        axis=1,
    ).drop_duplicates()

    db_pub_df = pd.concat(
        [
            db_datafusion["test_cas"],
            pd.DataFrame(
                pd.DataFrame(db_datafusion["pubchem2d"].values)
                .apply(
                    lambda x: x.str.replace("", " ").str.strip().str.split(" "), axis=1
                )[0]
                .to_list(),
                columns=["pub" + str(i) for i in range(1, 882)],
            ),
        ],
        axis=1,
    ).drop_duplicates()

    db_pub_invitro = pd.concat(
        [
            db_invitro["test_cas"],
            pd.DataFrame(
                pd.DataFrame(db_invitro["pubchem2d"].values)
                .apply(
                    lambda x: x.str.replace("", " ").str.strip().str.split(" "), axis=1
                )[0]
                .to_list(),
                columns=["pub" + str(i) for i in range(1, 882)],
            ),
        ],
        axis=1,
    ).drop_duplicates()

    db_mortality = db_mortality.merge(db_pub, on="test_cas").reset_index(drop=True)
    db_mortality.drop(columns="pubchem2d", inplace=True)

    db_datafusion = db_datafusion.merge(db_pub_df, on="test_cas").reset_index(drop=True)
    db_datafusion.drop(columns="pubchem2d", inplace=True)

    db_invitro = db_invitro.merge(db_pub_invitro, on="test_cas").reset_index(drop=True)
    db_invitro.drop(columns="pubchem2d", inplace=True)
    try:
        db_invitro = db_invitro.rename({"conc": "invitro_conc"}, axis=1)
    except:
        pass
    try:
        db_invitro = db_invitro.rename({"ec50": "invitro_conc"}, axis=1)
    except:
        pass
    db_raw = db_mortality.copy()

    # Ordinal encoding for categorical variables
    db_mortality, db_datafusion, db_invitro = encoding_categorical_invitro(
        categorical_both, db_raw, db_datafusion, db_invitro
    )

    db_mortality, db_datafusion = encoding_categorical(
        [item for item in categorical_columns if item not in categorical_both],
        db_mortality,
        db_datafusion,
    )

    # db_mortality, db_datafusion = encoding_categorical(
    #     categorical_columns, db_mortality, db_datafusion
    #     )
    # db_raw, db_invitro = encoding_categorical(categorical_both, db_raw, db_invitro)

    # concentration class labeling
    if encoding == "binary":
        db_mortality[conc_column] = np.where(
            db_mortality["conc1_mean"].values > encoding_value, 0, 1
        )
        db_invitro["invitro_label"] = np.where(
            db_invitro["invitro_conc"].values > 1, 0, 1
        )
        for ef in db_datafusion.effect.unique():
            conc = db_datafusion.loc[db_datafusion.effect == ef, "conc1_mean"]
            if fixed == "yes":
                print("fixed threshold")
                class_threshold = pd.read_csv("data/threshold.csv")
                threshold = class_threshold[class_threshold.effect == ef]
                db_datafusion.loc[db_datafusion.effect == ef, "conc1_mean"] = np.where(
                    conc > float(threshold.ths_binary), 0, 1
                )
            else:
                # print('unfixed threshold')
                db_datafusion.loc[db_datafusion.effect == ef, "conc1_mean"] = np.where(
                    conc > np.median(conc), 0, 1
                )
        # mortality always classified using 1
        # db_datafusion.loc[db_datafusion.effect == 'MOR', 'conc1_mean'] = np.where(conc > encoding_value, 0, 1)
    elif encoding == "multiclass":
        db_mortality[conc_column] = multiclass_encoding(
            db_mortality.conc_column.copy(), encoding_value
        )
        db_invitro["invitro_label"] = multiclass_encoding(
            db_invitro["invitro_conc"].copy(), encoding_value
        )
        for ef in db_datafusion.effect.unique():
            conc = db_datafusion.loc[db_datafusion.effect == ef, "conc1_mean"].copy()
            if fixed == "yes":
                print("fixed threshold")
                class_threshold = pd.read_csv("data/threshold.csv")
                threshold = class_threshold[class_threshold.effect == ef]
                db_datafusion.loc[
                    db_datafusion.effect == ef, "conc1_mean"
                ] = multiclass_encoding(
                    conc.values,
                    [
                        float(threshold.ths_1),
                        float(threshold.ths_2),
                        float(threshold.ths_3),
                        float(threshold.ths_4),
                    ],
                )
            else:
                # print("unfixed threshold")
                db_datafusion.loc[
                    db_datafusion.effect == ef, "conc1_mean"
                ] = multiclass_encoding(
                    conc.values, conc.quantile([0.2, 0.4, 0.6, 0.8]).values
                )
    return db_mortality, db_datafusion, db_invitro


# --------------------------------------------------------------------distance matrices-------------------
def hamming_matrix(X1, X2, cat_features):
    if X1.shape == X2.shape:
        if np.all(X1 == X2):
            return squareform(pdist(X1[cat_features], metric="hamming"))
    else:
        return cdist(X1[cat_features], X2[cat_features], metric="hamming")


def hamming_matrix_df(X1, X2, cat_features):
    return cdist(X1[cat_features], X2[cat_features], metric="hamming")


def euclidean_matrix_df(X1, X2, num_features):
    return cdist(X1[num_features], X2[num_features], metric="euclidean")


def euclidean_matrix(X1, X2, num_features):
    if X1.shape == X2.shape:
        if np.all(X1 == X2):
            return squareform(pdist(X1[num_features], metric="euclidean"))
    else:
        return cdist(X1[num_features], X2[num_features], metric="euclidean")


def pubchem2d_matrix(X1, X2):
    if X1.shape == X2.shape:
        if np.all(X1 == X2):
            return squareform(
                pdist(X1[X1.columns[X1.columns.str.contains("pub")]], metric="hamming")
            )
    else:
        return cdist(
            X1[X1.columns[X1.columns.str.contains("pub")]],
            X2[X2.columns[X2.columns.str.contains("pub")]],
            metric="hamming",
        )


def pubchem2d_matrix_df(X1, X2):
    df_1 = pd.DataFrame(
        [x.replace("", " ").strip().split(" ") for x in list(X1["pubchem2d"])]
    )

    df_2 = pd.DataFrame(
        [x.replace("", " ").strip().split(" ") for x in list(X2["pubchem2d"])]
    )

    return cdist(df_1, df_2, metric="hamming")


# euclidean matrix will always has 1 as parameter.
def matrix_combine(basic_mat, matrix_h, matrix_p, ah, ap):
    dist_matr = ah * matrix_h
    dist_matr += basic_mat
    dist_matr += ap * matrix_p
    dist_matr = pd.DataFrame(dist_matr)
    return dist_matr


def dist_matrix(X1, X2, non_categorical, categorical, ah, ap):

    matrix_h = hamming_matrix(X1, X2, categorical)
    dist_matr = ah * matrix_h
    del matrix_h
    basic_mat = euclidean_matrix(X1, X2, non_categorical)
    dist_matr += basic_mat
    del basic_mat
    matrix_p = pubchem2d_matrix(X1, X2)
    dist_matr += ap * matrix_p
    del matrix_p
    dist_matr = pd.DataFrame(dist_matr)

    return dist_matr


def cal_matrixs(X1, X2, categorical, non_categorical):
    matrices = {}
    basic_mat = euclidean_matrix(X1, X2, non_categorical)
    matrix_h = hamming_matrix(X1, X2, categorical)
    matrix_p = pubchem2d_matrix(X1, X2)
    matrices["euc"] = basic_mat
    matrices["hamming"] = matrix_h
    matrices["pubchem"] = matrix_p
    return matrices


def normalized_invitro_matrix(X_traintest, invitro_matrices, ah, ap):
    max_euc = pd.DataFrame(
        euclidean_matrix(X_traintest, X_traintest, non_categorical,)
    ).values.max()

    matrix_invitro_euc = pd.DataFrame(invitro_matrices["euc"])

    db_invitro_matrix_norm = pd.DataFrame(
        ah * invitro_matrices["hamming"]
        + ap * invitro_matrices["pubchem"]
        + matrix_invitro_euc.divide(max_euc).values
    )
    return db_invitro_matrix_norm


def get_traintest_matrices(matrices, fold, ah, ap):
    matrix_euc = pd.DataFrame(matrices["euc"])
    max_euc = matrix_euc.iloc[fold[0], fold[0]].values.max()

    distance_matrix = pd.DataFrame(
        ah * matrices["hamming"]
        + ap * matrices["pubchem"]
        + matrix_euc.divide(max_euc).values
    )

    train_matrix = distance_matrix.iloc[fold[0], fold[0]]
    test_matrix = distance_matrix.iloc[fold[1], fold[0]]
    return train_matrix, test_matrix


# ----------------------------------------------------------------------KNN model------------------------------------------------------------


def select_alpha(
    df_fishchem_tv,
    col_groups,
    X_train,
    Y_train,
    sequence_ham,
    categorical_columns,
    non_categorical_columns,
    leaf_ls,
    neighbors,
    encoding,
):
    print("calcaulting the matrix...", ctime())
    matrices = cal_matrixs(
        X_train, X_train, categorical_columns, non_categorical_columns
    )

    best_accs = 0
    num = 0
    print("Start selecting the best parameters....", ctime())
    for ah in sequence_ham:
        for ap in sequence_ham:
            for leaf in leaf_ls:
                for neigh in neighbors:
                    print(
                        "*" * 50,
                        num / (len(sequence_ham) ** 2 * len(leaf_ls) * len(neighbors)),
                        ctime(),
                        end="\r",
                    )
                    num = num + 1
                    df_output = KNN_model(
                        df_fishchem_tv,
                        col_groups,
                        matrices["hamming"],
                        matrices["pubchem"],
                        X_train,
                        Y_train,
                        leaf,
                        neigh,
                        ah,
                        ap,
                        encoding,
                    )

                    if df_output["accuracy"].mean() > best_accs + 0.0001:
                        print(
                            """New best params found! alpha_h:{}, alpha_p:{}, accs:{}, leaf:{},neighbor:{}""".format(
                                ah, ap, df_output["accuracy"].mean(), leaf, neigh
                            )
                        )

                        df_output.loc[0, "ah"] = ah
                        df_output.loc[0, "ap"] = ap
                        df_output.loc[0, "leaf"] = leaf
                        df_output.loc[0, "neighbor"] = neigh
                        best_results = df_output
                        best_accs = df_output["accuracy"].mean()

    return best_results


def KNN_model(
    df_fishchem_tv,
    col_groups,
    matrix_h,
    matrix_p,
    X,
    Y,
    leaf,
    neighbor,
    ah,
    ap,
    encoding,
    seed=25,
):

    group_kfold = GroupKFold(n_splits=5)

    accs = []
    sens = []
    precs = []
    specs = []
    f1 = []
    df_output = pd.DataFrame()
    for train_index, test_index in group_kfold.split(
        df_fishchem_tv, groups=df_fishchem_tv[col_groups]
    ):
        x_train = X.iloc[train_index]
        x_test = X.iloc[test_index]
        minmax = MinMaxScaler().fit(x_train[non_categorical])
        x_train[non_categorical] = minmax.transform(x_train.loc[:, non_categorical])
        x_test[non_categorical] = minmax.transform(x_test.loc[:, non_categorical])

        dist_matr_train = pd.DataFrame(
            ah * pd.DataFrame(matrix_h).iloc[train_index, train_index]
            + ap * pd.DataFrame(matrix_p).iloc[train_index, train_index]
            + euclidean_matrix(x_train, x_train, non_categorical)
        )
        dist_matr_test = pd.DataFrame(
            ah * pd.DataFrame(matrix_h).iloc[test_index, train_index]
            + ap * pd.DataFrame(matrix_p).iloc[test_index, train_index]
            + euclidean_matrix(x_test, x_train, non_categorical)
        )

        y_train = Y.iloc[train_index]
        y_test = Y.iloc[test_index]

        neigh = KNeighborsClassifier(
            n_neighbors=neighbor, metric="precomputed", leaf_size=int(leaf)
        )

        results = fit_and_predict(
            neigh,
            dist_matr_train,
            y_train.astype("int").ravel(),
            dist_matr_test,
            y_test,
            encoding,
        )
        df_output = pd.concat([df_output, results])

    return df_output

    #     neigh.fit(dist_matr_train, y_train.astype("int").ravel())
    #     y_pred = neigh.predict(dist_matr_test)

    #     if encoding == "binary":
    #         accs.append(accuracy_score(y_test, y_pred))
    #         sens.append(recall_score(y_test, y_pred))
    #         tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    #         specs.append(tn / (tn + fp))
    #         precs.append(precision_score(y_test, y_pred))
    #         f1.append(f1_score(y_test, y_pred))
    #     elif encoding == "multiclass":
    #         accs.append(accuracy_score(y_test, y_pred))
    #         sens.append(recall_score(y_test, y_pred, average="macro"))
    #         specs.append(np.nan)
    #         precs.append(precision_score(y_test, y_pred, average="macro"))
    #         f1.append(f1_score(y_test, y_pred, average="macro"))
    # return accs, sens, specs, precs, f1


# --------------------------------------------------------------------- RASAR---------------------------------------------------------


def find_neighbor(
    square_matr,
    matr_all_label,
    X_train,
    y_train,
    label,
    encoding="binary",
    exp="train",
    n_neighbors=2,
):

    idx_neigh = pd.DataFrame()
    dist = pd.DataFrame()
    knn = KNeighborsClassifier(
        metric="precomputed", leaf_size=30, n_neighbors=n_neighbors
    )

    if len(y_train[y_train == label]) == 1:
        idx_neigh = np.array(square_matr.index * matr_all_label.shape[0])
        dist = pd.DataFrame(np.array([0] * matr_all_label.shape[0]))
    else:
        knn.fit(square_matr, y_train[y_train == label])
        neigh = knn.kneighbors(matr_all_label, return_distance=True)

        for i in range(n_neighbors):
            idx_neigh["idx_neigh" + str(label) + str(i)] = pd.DataFrame(neigh[1])[
                i
            ].apply(lambda x: X_train.iloc[y_train == label].iloc[x].name)
            dist["dist_neigh" + str(label) + str(i)] = pd.DataFrame(neigh[0])[i]

        # if the nearest point index equals itself, then replacing it with the second nearst point
        # Distance from the Nearest Neighbor that is NOT itself
        if exp == "train":
            ls = X_train.index == idx_neigh["idx_neigh" + str(label) + str(0)]
            for i in range(n_neighbors - 1):
                idx_neigh["idx_neigh" + str(label) + str(i)][ls] = idx_neigh[
                    "idx_neigh" + str(label) + str(i + 1)
                ][ls].values
                dist["dist_neigh" + str(label) + str(i)][ls] = dist[
                    "dist_neigh" + str(label) + str(i + 1)
                ][ls].values

        idx_neigh.drop(
            columns="idx_neigh" + str(label) + str(n_neighbors - 1), inplace=True
        )
        dist.drop(
            columns="dist_neigh" + str(label) + str(n_neighbors - 1), inplace=True
        )

    return idx_neigh, dist


def find_neighbor_invitro(
    rasar_train, rasar_test, db_invitro, db_mortality_train, db_mortality_test, ah, ap
):
    invitro_matrix = dist_matrix(
        db_invitro, db_invitro, non_categorical, categorical_both, ah, ap
    )
    knn = KNeighborsClassifier(metric="precomputed", n_neighbors=1)
    knn.fit(pd.DataFrame(invitro_matrix), np.repeat(0, invitro_matrix.shape[0]))

    train_invitro_matrix = dist_matrix(
        db_mortality_train, db_invitro, non_categorical, categorical_both, ah, ap
    )
    neigh = knn.kneighbors(train_invitro_matrix, return_distance=True)
    # print(rasar_train.shape)
    # print(len(neigh[0]))
    rasar_train[["invitro_conc"]] = (
        pd.DataFrame(neigh[1])
        .apply(lambda x: db_invitro["invitro_conc"][x])
        .reset_index(drop=True)
    )
    rasar_train[["invitro_label"]] = (
        pd.DataFrame(neigh[1])
        .apply(lambda x: db_invitro["invitro_label"][x])
        .reset_index(drop=True)
    )
    rasar_train[["invitro_dist"]] = pd.DataFrame(neigh[0])
    # print(len(pd.DataFrame(neigh[0]).values))

    test_invitro_matrix = dist_matrix(
        db_mortality_test, db_invitro, non_categorical, categorical_both, ah, ap
    )
    neigh = knn.kneighbors(test_invitro_matrix, return_distance=True)

    rasar_test[["invitro_conc"]] = (
        pd.DataFrame(neigh[1])
        .apply(lambda x: db_invitro["invitro_conc"][x])
        .reset_index(drop=True)
    )
    rasar_test[["invitro_label"]] = (
        pd.DataFrame(neigh[1])
        .apply(lambda x: db_invitro["invitro_label"][x])
        .reset_index(drop=True)
    )
    rasar_test[["invitro_dist"]] = pd.DataFrame(neigh[0])
    # print(len(pd.DataFrame(neigh[0]).values))

    return rasar_train, rasar_test


def take_per_row_strided(A, indx, num_elem=2):
    # print("A:",A)
    # print("indx:",indx)
    m, n = A.shape
    # A.shape = (-1)
    A = A.reshape(-1)
    s0 = A.strides[0]
    l_indx = indx + n * np.arange(len(indx))
    # print(l_indx)
    out = as_strided(A, (len(A) - int(num_elem) + 1, num_elem), (s0, s0))[l_indx]
    A.shape = m, n

    # print("out:",out)
    return out


def cal_s_rasar(
    train_distance_matrix,
    test_distance_matrix,
    y_train,
    n_neighbors=1,
    encoding="binary",
):
    """calculate the nearest points in each class for training experiments with alphas"""
    df_rasar_train = pd.DataFrame()
    df_rasar_test = pd.DataFrame()
    if encoding == "binary":
        labels = [0, 1]
    elif encoding == "multiclass":
        labels = [0, 1, 2, 3, 4]

    for i in labels:
        dist_matr_train_train_i = train_distance_matrix.iloc[:, y_train == i]
        values = dist_matr_train_train_i.values
        values.sort(axis=1)
        indx = (dist_matr_train_train_i == 0).astype("int64").sum(axis=1).values
        disti = pd.DataFrame(take_per_row_strided(values, indx, n_neighbors))
        df_rasar_train = pd.concat([disti, df_rasar_train], axis=1)

        dist_matr_test_test_i = test_distance_matrix.iloc[:, y_train == i]
        values = dist_matr_test_test_i.values
        values.sort(axis=1)
        indx = (dist_matr_test_test_i == 0).astype("int64").sum(axis=1).values
        disti = pd.DataFrame(take_per_row_strided(values, indx, n_neighbors))
        df_rasar_test = pd.concat([disti, df_rasar_test], axis=1)
    df_rasar_train.columns = range(df_rasar_train.shape[1])
    df_rasar_train.columns = [str(x) for x in df_rasar_train.columns]
    df_rasar_test.columns = range(df_rasar_test.shape[1])
    df_rasar_test.columns = [str(x) for x in df_rasar_test.columns]
    return df_rasar_train, df_rasar_test


def get_vitroinfo(df, X, index, invitro_form):

    conc = X.iloc[index, :].invitro_conc.reset_index(drop=True)
    label = X.iloc[index, :].invitro_label.reset_index(drop=True)

    if invitro_form == "number":
        df["invitro_conc"] = conc
    elif invitro_form == "label":
        df["invitro_label"] = label
    elif invitro_form == "both":
        df["invitro_conc"] = conc
        df["invitro_label"] = label
    elif invitro_form == "label_half":
        df["invitro_label_half"] = X.iloc[index, :].invitro_label_half.reset_index(
            drop=True
        )
    elif invitro_form == "both_half":
        df["invitro_conc"] = conc
        df["invitro_label_half"] = X.iloc[index, :].invitro_label_half.reset_index(
            drop=True
        )
    elif invitro_form == "label_reserved":
        df["invitro_label_reserved"] = X.iloc[
            index, :
        ].invitro_label_reserved.reset_index(drop=True)
    elif invitro_form == "both_reserved":
        df["invitro_conc"] = conc
        df["invitro_label_reserved"] = X.iloc[
            index, :
        ].invitro_label_reserved.reset_index(drop=True)
    elif invitro_form == "label_half_reserved":
        df["invitro_label_half_reserved"] = X.iloc[
            index, :
        ].invitro_label_half_reserved.reset_index(drop=True)
    elif invitro_form == "both_half_reserved":
        df["invitro_conc"] = conc
        df["invitro_label_half_reserved"] = X.iloc[
            index, :
        ].invitro_label_half_reserved.reset_index(drop=True)
    return df


def find_nearest_vitro(df, db_invitro, invitro_matrix, index, invitro_form):

    dist = np.array(invitro_matrix.iloc[index, :].min(axis=1))

    ls = np.array(invitro_matrix.iloc[index, :].idxmin(axis=1))

    conc = db_invitro.iloc[ls, :].invitro_conc.reset_index(drop=True)
    label = db_invitro.iloc[ls, :].invitro_label.reset_index(drop=True)

    df["invitro_dist"] = dist
    if invitro_form == "number":
        df["invitro_conc"] = np.array(conc)
    elif invitro_form == "label":
        df["invitro_label"] = np.array(label)
    elif invitro_form == "both":
        df["invitro_conc"] = np.array(conc)
        df["invitro_label"] = np.array(label)
    elif invitro_form == "label_half":
        label_half = db_invitro.iloc[ls, :].invitro_label_half.reset_index(drop=True)
        df["invitro_label_half"] = np.array(label_half)
    elif invitro_form == "both_half":
        label_half = db_invitro.iloc[ls, :].invitro_label_half.reset_index(drop=True)
        df["invitro_conc"] = np.array(conc)
        df["invitro_label_half"] = np.array(label_half)
    return df


def RASAR_simple(
    df_fishchem_tv,
    col_groups,
    matrices,
    ah,
    ap,
    X,
    Y,
    db_invitro_matrices,
    n_neighbors=int(1),
    invitro="False",
    invitro_form="number",
    db_invitro="noinvitro",
    encoding="binary",
    model=RandomForestClassifier(random_state=0, n_estimators=200),
):

    group_kfold = GroupKFold(n_splits=5)

    list_df_output = []
    for train_index, test_index in group_kfold.split(
        df_fishchem_tv, groups=df_fishchem_tv[col_groups]
    ):

        dist_matr_train, dist_matr_test = get_traintest_matrices(
            matrices, [train_index, test_index], ah, ap
        )

        y_train = Y[train_index]
        y_test = Y[test_index]

        train_rf, test_rf = cal_s_rasar(
            dist_matr_train, dist_matr_test, y_train, n_neighbors, encoding,
        )

        if invitro == "own":
            train_rf = pd.DataFrame()
            test_rf = pd.DataFrame()

        if invitro != "False":
            if str(db_invitro) == "overlap":
                train_rf = get_vitroinfo(train_rf, X, train_index, invitro_form)
                test_rf = get_vitroinfo(test_rf, X, test_index, invitro_form)
            else:
                matrix_euc = pd.DataFrame(matrices["euc"])
                max_euc = matrix_euc.iloc[train_index, train_index].values.max()

                db_invitro_matrix_normed = pd.DataFrame(
                    ah * db_invitro_matrices["hamming"]
                    + ap * db_invitro_matrices["pubchem"]
                    + pd.DataFrame(db_invitro_matrices["euc"]).divide(max_euc).values
                )
                train_rf = find_nearest_vitro(
                    train_rf,
                    db_invitro,
                    db_invitro_matrix_normed,
                    train_index,
                    invitro_form,
                )
                test_rf = find_nearest_vitro(
                    test_rf,
                    db_invitro,
                    db_invitro_matrix_normed,
                    test_index,
                    invitro_form,
                )

        df_score = fit_and_predict(model, train_rf, y_train, test_rf, y_test, encoding,)
        df_score["neighbors"] = n_neighbors
        df_score["ah"] = ah
        df_score["ap"] = ap

        list_df_output.append(df_score)

    del y_train

    df_output = pd.concat(list_df_output, axis=0)
    return df_output


# --------------------------------------------------------------------- datafusion---------------------------------------------------------


def find_similar_exp(db_mortality, db_datafusion_rasar, db_endpoint, effect, encoding):
    try:
        temp = pd.merge(
            db_mortality.reset_index(drop=True),
            db_endpoint[db_endpoint.effect == effect],
            on=comparing,
            how="left",
        )
        # temp = temp[temp.atom_number_y.notnull()]
        temp = temp[
            temp[np.setdiff1d(db_mortality.columns, comparing)[0] + "_y"].notnull()
        ]
        temp = pd.DataFrame(
            temp.groupby(comparing)["conc1_mean"].agg(pd.Series.mode)
        ).reset_index()
        temp = pd.merge(
            db_mortality.reset_index(drop=True), temp, on=comparing, how="left"
        )
        if encoding == "binary":
            temp["conc1_mean"] = np.where(
                temp["conc1_mean"] == 0, -1, (np.where(temp["conc1_mean"] == 1, 1, 0))
            )
        elif encoding == "multiclass":
            temp["conc1_mean"] = temp["conc1_mean"].fillna("Unknown")
    except:
        if encoding == "binary":
            temp = pd.DataFrame(
                0, index=np.arange(len(db_datafusion_rasar)), columns=["conc1_mean"]
            )
        elif encoding == "multiclass":
            temp = pd.DataFrame(
                "Unknown",
                index=np.arange(len(db_datafusion_rasar)),
                columns=["conc1_mean"],
            )

    return temp


def find_datafusion_neighbor(
    db_datafusion_rasar_train,
    db_datafusion_rasar_test,
    db_datafusion,
    db_datafusion_matrix,
    train_index,
    test_index,
    effect,
    endpoint,
    encoding="binary",
):
    if encoding == "binary":
        label = [0, 1]
    elif encoding == "multiclass":
        label = [0, 1, 2, 3, 4]

    for a in label:
        db_end_eff = db_datafusion.loc[
            (db_datafusion.effect == effect)
            & (db_datafusion.conc1_mean == a)
            & (db_datafusion.endpoint == endpoint)
        ]
        if len(db_end_eff) == 0:
            continue
        else:
            train_test_matrix = db_datafusion_matrix.iloc[
                train_index,
                np.array(
                    (db_datafusion.effect == effect)
                    & (db_datafusion.conc1_mean == a)
                    & (db_datafusion.endpoint == endpoint)
                ).nonzero()[0],
            ]
            train_test_matrix = train_test_matrix.reset_index(drop=True)
            train_test_matrix.columns = range(train_test_matrix.shape[1])
            col_name = endpoint + "_" + effect + "_" + str(a)

            db_datafusion_rasar_train[col_name] = np.array(
                train_test_matrix.min(axis=1)
            )

            test_test_matrix = db_datafusion_matrix.iloc[
                test_index,
                np.array(
                    (db_datafusion.effect == effect)
                    & (db_datafusion.conc1_mean == a)
                    & (db_datafusion.endpoint == endpoint)
                ).nonzero()[0],
            ]
            test_test_matrix = test_test_matrix.reset_index(drop=True)
            test_test_matrix.columns = range(test_test_matrix.shape[1])

            db_datafusion_rasar_test[col_name] = np.array(test_test_matrix.min(axis=1))

    return db_datafusion_rasar_train, db_datafusion_rasar_test


def cal_df_rasar(
    train_index,
    test_index,
    db_mortality_train,
    db_mortality_test,
    db_datafusion,
    db_datafusion_matrix,
    train_label,
    train_effect,
    encoding,
):
    db_datafusion_rasar_train = pd.DataFrame()
    db_datafusion_rasar_test = pd.DataFrame()

    for endpoint in db_datafusion.endpoint.unique():

        db_endpoint = db_datafusion[db_datafusion.endpoint == endpoint]
        for effect in db_endpoint.effect.unique():
            if (str(effect) == train_effect) & (str(endpoint) in train_label):
                continue
            else:
                (
                    db_datafusion_rasar_train,
                    db_datafusion_rasar_test,
                ) = find_datafusion_neighbor(
                    db_datafusion_rasar_train,
                    db_datafusion_rasar_test,
                    db_datafusion,
                    db_datafusion_matrix,
                    train_index,
                    test_index,
                    effect,
                    endpoint,
                    encoding,
                )

                # FINDING LABELS
                col_name = endpoint + "_" + effect + "_label"
                temp = find_similar_exp(
                    db_mortality_train,
                    db_datafusion_rasar_train,
                    db_endpoint,
                    effect,
                    encoding,
                )

                db_datafusion_rasar_train[col_name] = temp["conc1_mean"]

                temp = find_similar_exp(
                    db_mortality_test,
                    db_datafusion_rasar_test,
                    db_endpoint,
                    effect,
                    encoding,
                )

                db_datafusion_rasar_test[col_name] = temp["conc1_mean"]

    return db_datafusion_rasar_train, db_datafusion_rasar_test


def cv_datafusion_rasar(
    matrices,
    matrices_df,
    matrices_invitro,
    ah,
    ap,
    X,
    Y,
    db_datafusion,
    db_invitro,
    train_endpoint,
    train_effect,
    df_fishchem_tv,
    col_groups,
    model=RandomForestClassifier(random_state=10),
    n_neighbors=1,
    invitro=False,
    invitro_form="both",
    encoding="binary",
    wmor=True,
):

    group_kfold = GroupKFold(n_splits=5)

    list_df_output = []
    for train_index, test_index in group_kfold.split(
        df_fishchem_tv, groups=df_fishchem_tv[col_groups]
    ):
        matrix_euc = pd.DataFrame(matrices["euc"])
        max_euc = matrix_euc.iloc[train_index, train_index].values.max()

        distance_matrix = pd.DataFrame(
            ah * matrices["hamming"]
            + ap * matrices["pubchem"]
            + matrix_euc.divide(max_euc).values
        )
        db_datafusion_matrix = pd.DataFrame(
            ah * matrices_df["hamming"]
            + ap * matrices_df["pubchem"]
            + pd.DataFrame(matrices_df["pubchem"]).divide(max_euc).values
        )

        dist_matr_train = distance_matrix.iloc[train_index, train_index]
        dist_matr_test = distance_matrix.iloc[test_index, train_index]

        X_copy = X.copy()

        new_db_datafusion = db_datafusion.copy()

        y_train = Y[train_index]
        y_test = Y[test_index]
        if wmor:
            s_rasar_train, s_rasar_test = cal_s_rasar(
                dist_matr_train, dist_matr_test, y_train, n_neighbors, encoding
            )
        else:
            s_rasar_train, s_rasar_test = None, None

        df_rasar_train, df_rasar_test = cal_df_rasar(
            train_index,
            test_index,
            X_copy.iloc[train_index],
            X_copy.iloc[test_index],
            new_db_datafusion,
            db_datafusion_matrix,
            train_endpoint,
            train_effect,
            encoding,
        )

        train_rf = pd.concat([s_rasar_train, df_rasar_train], axis=1)
        test_rf = pd.concat([s_rasar_test, df_rasar_test], axis=1)
        del (
            dist_matr_train,
            dist_matr_test,
            s_rasar_train,
            s_rasar_test,
            df_rasar_train,
            df_rasar_test,
        )
        if invitro == "own":
            train_rf = pd.DataFrame()
            test_rf = pd.DataFrame()

        if invitro != "False":
            if str(db_invitro) == "overlap":
                train_rf = get_vitroinfo(train_rf, X_copy, train_index, invitro_form)
                test_rf = get_vitroinfo(test_rf, X_copy, test_index, invitro_form)
            else:
                matrices_invitro_normed = pd.DataFrame(
                    ah * matrices_invitro["hamming"]
                    + ap * matrices_invitro["pubchem"]
                    + pd.DataFrame(matrices_invitro["euc"]).divide(max_euc).values
                )
                train_rf = find_nearest_vitro(
                    train_rf,
                    db_invitro,
                    matrices_invitro_normed,
                    train_index,
                    invitro_form,
                )
                test_rf = find_nearest_vitro(
                    test_rf,
                    db_invitro,
                    matrices_invitro_normed,
                    test_index,
                    invitro_form,
                )

        if encoding == "binary":
            df_score = fit_and_predict(
                model, train_rf, y_train, test_rf, y_test, encoding,
            )
            df_score.loc[0, ["ah", "ap"]] = ah, ap
            list_df_output.append(df_score)
        elif encoding == "multiclass":

            train_rf.loc[:, "target"] = y_train
            test_rf.loc[:, "target"] = y_test

            train_rf_h2o = h2o.H2OFrame(train_rf)
            test_rf_h2o = h2o.H2OFrame(test_rf)

            for col in train_rf.columns:
                if "label" in col:
                    train_rf_h2o[col] = train_rf_h2o[col].asfactor()
                    test_rf_h2o[col] = test_rf_h2o[col].asfactor()

            train_rf_h2o["target"] = train_rf_h2o["target"].asfactor()
            test_rf_h2o["target"] = test_rf_h2o["target"].asfactor()

            model.train(y="target", training_frame=train_rf_h2o)
            y_pred = model.predict(test_rf_h2o).as_data_frame()["predict"]

            df_score = pd.DataFrame()
            df_score.loc[0, "accuracy"] = accuracy_score(y_test, y_pred)
            df_score.loc[0, "recall"] = recall_score(y_test, y_pred, average="macro")
            df_score.loc[0, "specificity"] = np.nan
            df_score.loc[0, "f1"] = f1_score(y_test, y_pred, average="macro")
            df_score.loc[0, "precision"] = precision_score(
                y_test, y_pred, average="macro"
            )
            df_score.loc[0, ["ah", "ap"]] = ah, ap

            list_df_output.append(df_score)

        del train_rf, test_rf

    df_output = pd.concat(list_df_output, axis=0)
    return df_output


# ---------------------------------------------------------------------datafusion_multiclass--------------------------------------------------


def cal_df_rasar_multiclass(
    db_mortality_train,
    db_mortality_test,
    db_datafusion,
    train_label,
    train_effect,
    ah,
    ap,
):

    db_datafusion_rasar_train = pd.DataFrame()
    db_datafusion_rasar_test = pd.DataFrame()

    for endpoint in db_datafusion.endpoint.unique():

        db_endpoint = db_datafusion[db_datafusion.endpoint == endpoint]

        for effect in db_endpoint.effect.unique():
            if (str(effect) == train_effect) & (str(endpoint) in train_label):
                continue
            else:
                for i in range(1, 6):
                    (
                        db_datafusion_rasar_train,
                        db_datafusion_rasar_test,
                    ) = find_datafusion_neighbor(
                        db_datafusion_rasar_train,
                        db_datafusion_rasar_test,
                        db_endpoint,
                        db_mortality_train,
                        db_mortality_test,
                        effect,
                        endpoint,
                        ah,
                        ap,
                        [i],
                    )

                db_datafusion_rasar_train[
                    endpoint + "_" + effect + "_label"
                ] = db_mortality_train.apply(
                    lambda x: find_similar_exp(
                        x,
                        db_endpoint[db_endpoint.effect == effect],
                        comparing,
                        "multiclass",
                    ),
                    axis=1,
                ).reset_index(
                    drop=True
                )
                db_datafusion_rasar_test[
                    endpoint + "_" + effect + "_label"
                ] = db_mortality_test.apply(
                    lambda x: find_similar_exp(
                        x,
                        db_endpoint[db_endpoint.effect == effect],
                        comparing,
                        "multiclass",
                    ),
                    axis=1,
                ).reset_index(
                    drop=True
                )

    return db_datafusion_rasar_train, db_datafusion_rasar_test


def cv_datafusion_rasar_multiclass(
    db_datafusion_matrix,
    distance_matrix,
    db_invitro_matrix,
    X,
    y,
    db_datafusion,
    db_invitro,
    train_label,
    train_effect,
    final_model=False,
    n_neighbors=2,
    invitro=False,
    invitro_form="both",
    encoding="multiclass",
):
    # h2o.init(max_mem_size="20g")
    h2o.init()
    h2o.no_progress()

    # print('Start CV...', ctime())
    kf = KFold(n_splits=5, shuffle=True, random_state=10)
    kf = KFold(n_splits=3, shuffle=True, random_state=12)
    accs = []
    rmse = []
    sens = []
    precs = []
    f1 = []
    result_pred = pd.DataFrame()
    for train_index, test_index in kf.split(distance_matrix):

        dist_matr_train = distance_matrix.iloc[train_index, train_index]
        dist_matr_test = distance_matrix.iloc[test_index, train_index]
        y_train = y[train_index]
        y_test = y[test_index]

        # print('Train and test...')
        simple_rasar_train, simple_rasar_test = cal_s_rasar(
            dist_matr_train, dist_matr_test, y_train, n_neighbors, encoding
        )
        # print('Finish simple dataset')
        # print('Start datafusion dataset.')
        datafusion_rasar_train, datafusion_rasar_test = cal_df_rasar(
            train_index,
            test_index,
            X.iloc[train_index],
            X.iloc[test_index],
            db_datafusion,
            db_datafusion_matrix,
            train_label,
            train_effect,
            encoding,
        )
        del dist_matr_train, dist_matr_test

        train_rf = pd.concat([simple_rasar_train, datafusion_rasar_train], axis=1)
        test_rf = pd.concat([simple_rasar_test, datafusion_rasar_test], axis=1)

        if str(db_invitro) == "overlap":
            if (invitro != "False") & (invitro_form == "number"):
                train_rf["invitro_conc"] = X.invitro_conc[train_index].reset_index(
                    drop=True
                )
                test_rf["invitro_conc"] = X.invitro_conc[test_index].reset_index(
                    drop=True
                )

            elif (invitro != "False") & (invitro_form == "label"):
                train_rf["invitro_label"] = X.invitro_label[train_index].reset_index(
                    drop=True
                )
                test_rf["invitro_label"] = X.invitro_label[test_index].reset_index(
                    drop=True
                )
            elif (invitro != "False") & (invitro_form == "both"):
                train_rf["invitro_conc"] = X.invitro_conc[train_index].reset_index(
                    drop=True
                )
                test_rf["invitro_conc"] = X.invitro_conc[test_index].reset_index(
                    drop=True
                )
                train_rf["invitro_label"] = X.invitro_label[train_index].reset_index(
                    drop=True
                )
                test_rf["invitro_label"] = X.invitro_label[test_index].reset_index(
                    drop=True
                )
        else:
            if invitro == "own":
                train_rf = pd.DataFrame()
                test_rf = pd.DataFrame()

            if (invitro != "False") & (invitro_form == "number"):
                dist = np.array(db_invitro_matrix.iloc[train_index, :].min(axis=1))
                ls = np.array(db_invitro_matrix.iloc[train_index, :].idxmin(axis=1))
                conc = db_invitro["invitro_conc"][ls]
                dist = db_invitro_matrix.lookup(
                    pd.Series(ls).index, pd.Series(ls).values
                )
                train_rf["invitro_conc"] = np.array(conc)
                train_rf["invitro_dist"] = dist

                dist = np.array(db_invitro_matrix.iloc[test_index, :].min(axis=1))
                ls = np.array(db_invitro_matrix.iloc[test_index, :].idxmin(axis=1))
                conc = db_invitro["invitro_conc"][ls]
                dist = db_invitro_matrix.lookup(
                    pd.Series(ls).index, pd.Series(ls).values
                )
                test_rf["invitro_conc"] = np.array(conc)
                test_rf["invitro_dist"] = dist

            elif (invitro != "False") & (invitro_form == "label"):
                dist = np.array(db_invitro_matrix.iloc[train_index, :].min(axis=1))
                ls = np.array(db_invitro_matrix.iloc[train_index, :].idxmin(axis=1))
                label = db_invitro["invitro_label"][ls]
                dist = db_invitro_matrix.lookup(
                    pd.Series(ls).index, pd.Series(ls).values
                )
                train_rf["invitro_label"] = np.array(label)
                train_rf["invitro_dist"] = dist

                dist = np.array(db_invitro_matrix.iloc[test_index, :].min(axis=1))
                ls = np.array(db_invitro_matrix.iloc[test_index, :].idxmin(axis=1))
                label = db_invitro["invitro_label"][ls]
                dist = db_invitro_matrix.lookup(
                    pd.Series(ls).index, pd.Series(ls).values
                )
                test_rf["invitro_label"] = np.array(label)
                test_rf["invitro_dist"] = dist

            elif (invitro != "False") & (invitro_form == "both"):

                dist = np.array(db_invitro_matrix.iloc[train_index, :].min(axis=1))
                ls = np.array(db_invitro_matrix.iloc[train_index, :].idxmin(axis=1))
                conc = db_invitro["invitro_conc"][ls]
                label = db_invitro["invitro_label"][ls]
                train_rf["invitro_conc"] = np.array(conc)
                train_rf["invitro_label"] = np.array(label)
                train_rf["invitro_dist"] = dist

                dist = np.array(db_invitro_matrix.iloc[test_index, :].min(axis=1))
                ls = np.array(db_invitro_matrix.iloc[test_index, :].idxmin(axis=1))
                conc = db_invitro["invitro_conc"][ls]
                label = db_invitro["invitro_label"][ls]
                test_rf["invitro_conc"] = np.array(conc)
                test_rf["invitro_label"] = np.array(label)
                test_rf["invitro_dist"] = dist

        train_rf.loc[:, "target"] = y_train
        test_rf.loc[:, "target"] = y_test

        train_rf_h2o = h2o.H2OFrame(train_rf)
        test_rf_h2o = h2o.H2OFrame(test_rf)

        for col in train_rf.columns:
            if "label" in col:
                train_rf_h2o[col] = train_rf_h2o[col].asfactor()
                test_rf_h2o[col] = test_rf_h2o[col].asfactor()

        train_rf_h2o["target"] = train_rf_h2o["target"].asfactor()
        test_rf_h2o["target"] = test_rf_h2o["target"].asfactor()

        # print('done...', ctime(), 'model...')

        if not final_model:
            rfc = H2ORandomForestEstimator(
                categorical_encoding="one_hot_explicit", seed=123
            )
        else:
            ### TO DO TO DO TO DO TO DO ### ### ### ### ### ### ### ### ### ###  TO DO  TO DO  TO DO TO DO TO DO TO DO TO DO
            rfc = (
                H2ORandomForestEstimator()
            )  ### TO DO TO DO TO DO TO DO TO DO TO DO TO DO TO DO TO DO TO DO TO DO TO DO
            ### TO DO TO DO TO DO TO DO ### ### ### ### ### ### ### ### ### ### TO DO TO DO TO DO TO DO TO DO TO DO

        rfc.train(y="target", training_frame=train_rf_h2o)
        y_pred = rfc.predict(test_rf_h2o).as_data_frame()["predict"]

        accs.append(accuracy_score(y_test, y_pred))
        rmse.append(sqrt(mean_squared_error(y_test, y_pred)))
        sens.append(recall_score(y_test, y_pred, average="macro"))
        precs.append(precision_score(y_test, y_pred, average="macro"))
        f1.append(f1_score(y_test, y_pred, average="macro"))
        del y_pred, train_rf, test_rf, train_rf_h2o, test_rf_h2o
    # h2o.cluster().shutdown()
    results = {}
    results["avg_accs"] = np.mean(accs)
    results["se_accs"] = sem(accs)
    results["avg_rmse"] = np.mean(rmse)
    results["se_rmse"] = sem(rmse)
    results["avg_sens"] = np.mean(sens)
    results["se_sens"] = sem(sens)
    results["avg_precs"] = np.mean(precs)
    results["se_precs"] = sem(precs)
    results["avg_f1"] = np.mean(f1)
    results["se_f1"] = sem(f1)
    results["fold"] = kf
    results["model"] = rfc
    return results

