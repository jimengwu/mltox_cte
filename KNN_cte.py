from helper_model import *
import numpy as np
from time import ctime
import argparse
import sys
import os


def getArguments():
    parser = argparse.ArgumentParser(description="Running KNN_model for datasets.")
    parser.add_argument("-i", "--input", dest="inputFile", required=True)
    parser.add_argument("-e", "--encoding", dest="encoding", default="binary")
    parser.add_argument(
        "-l", "--leaf_ls", dest="leaf_list", required=True, nargs="+", type=int
    )
    parser.add_argument(
        "-n", "--neighbors", dest="neighbors", required=True, nargs="+", type=int
    )
    parser.add_argument("-o", "--output", dest="outputFile", default="binary.txt")
    return parser.parse_args()


rand = random.randrange(1, 100)

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
    "invitro_label",
]
invitro = True
if invitro:
    categorical = ["class", "tax_order", "family", "genus", "species"]
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
    # "invitro_conc"
]
args = getArguments()

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
    encoding_value=encoding_value,
    seed=42,
)
print("finish loaded.", ctime())
# db_mortality = db_mortality[:300]

df_fishchem = db_mortality[["fish", "test_cas"]]
test_size = 0.2
col_groups = "test_cas"

trainvalid_idx, test_idx = get_grouped_train_test_split(
    df_fishchem, test_size, col_groups
)
df_fishchem_tv = df_fishchem.iloc[trainvalid_idx, :].reset_index(drop=True)
X_train = (
    db_mortality.drop(columns="conc1_mean")
    .iloc[trainvalid_idx, :]
    .reset_index(drop=True)
)
if "conc1_mean" in list(X_train.columns):
    print("yes")
else:
    print("no")

X_test = (
    db_mortality.drop(columns="conc1_mean").iloc[test_idx, :].reset_index(drop=True)
)

Y_train = db_mortality.iloc[trainvalid_idx, :].conc1_mean
Y_test = db_mortality.iloc[test_idx, :].conc1_mean


# X_train, X_test, Y_train, Y_test = train_test_split(
#     X, Y, test_size=0.2, random_state=42
# )

# using 5-fold cross validation to choose the alphas with best accuracy

# sequence_alpha = np.concatenate([
#     np.logspace(-1.5, -1, 20),
# np.logspace(-3, 0, 25),
# np.linspace(1.5, 10, 5),
# np.logspace(-2, 1, 30),
# np.logspace(-0.1, 1, 30),
# np.logspace(-0.8, 0, 10)
# ])

# sequence_alpha = [0.06723357536499334, 3.039195382313198]
# [0.5623, 0.0749, 0.2212, 1.7433, 0.0833, 0.9459]
# sequence_alpha = [0.01268961003167922,0.3562247890262442, 0.08531678524172806, 4.893900918477494, 0.06723357536499334, 3.039195382313198]
# np.concatenate(
# [np.logspace(-3, 0, 25), np.linspace(1.5, 10, 5)])
sequence_alpha = np.logspace(-5, 0, 30)
sequence_alpha = np.logspace(-2, 1, 30)
# sequence_alpha = np.concatenate(
#     [
#         np.logspace(-5, 0, 30),
#         np.array(
#             [
#                 0.01268961003167922,
#                 0.3562247890262442,
#                 0.08531678524172806,
#                 4.893900918477494,
#                 0.06723357536499334,
#                 3.039195382313198,
#             ]
#         ),
#     ]
# )

best_alpha_h, best_alpha_p, best_leaf, best_neighbor, best_results = select_alpha(
    df_fishchem_tv,
    col_groups,
    X_train,
    Y_train,
    sequence_alpha,
    categorical,
    non_categorical,
    args.leaf_list,
    args.neighbors,
    encoding,
)
print(ctime())

# validate on the test dataset
minmax = MinMaxScaler().fit(X_train[non_categorical])
X_train[non_categorical] = minmax.transform(X_train.loc[:, non_categorical])
X_test[non_categorical] = minmax.transform(X_test.loc[:, non_categorical])

matrix = dist_matrix(
    X_test, X_train, non_categorical, categorical, best_alpha_h, best_alpha_p
)
matrix_train = dist_matrix(
    X_train, X_train, non_categorical, categorical, best_alpha_h, best_alpha_p
)
neigh = KNeighborsClassifier(
    n_neighbors=best_neighbor, metric="precomputed", leaf_size=best_leaf
)
neigh.fit(matrix_train, Y_train.astype("int").ravel())
y_pred = neigh.predict(matrix)

if encoding == "binary":

    accs = accuracy_score(Y_test, y_pred)
    sens = recall_score(Y_test, y_pred, average="macro")
    tn, fp, fn, tp = confusion_matrix(Y_test, y_pred, labels=[0, 1]).ravel()
    specs = tn / (tn + fp)
    precs = precision_score(Y_test, y_pred, average="macro")
    f1 = f1_score(Y_test, y_pred, average="macro")

elif encoding == "multiclass":
    accs = accuracy_score(Y_test, y_pred)
    sens = recall_score(Y_test, y_pred, average="macro")
    specs = np.nan
    precs = precision_score(Y_test, y_pred, average="macro")
    f1 = f1_score(Y_test, y_pred, average="macro")


print(
    "Accuracy: ",
    accs,
    "se:",
    best_results["se_accs"],
    "\n Sensitivity:",
    sens,
    "se:",
    best_results["se_sens"],
    "\n Specificity",
    specs,
    "se:",
    best_results["se_specs"],
    "\n Precision",
    precs,
    "se:",
    best_results["se_precs"],
    "\n F1 score:",
    f1,
    "se:",
    best_results["se_f1"],
)

# saving the information into a file
info = []
info.append(
    """The best params were alpha_h:{}, alpha_p:{} ,leaf:{},neighbor:{}""".format(
        best_alpha_h, best_alpha_p, best_leaf, best_neighbor
    )
)
info.append(
    """Accuracy:  {}, Se.Accuracy:  {} 
		\nSensitivity:  {}, Se.Sensitivity: {}
        \nSpecificity:  {}, Se.Specificity:{}
		\nPrecision:  {}, Se.Precision: {}
		\nf1_score:{}, Se.f1_score:{}""".format(
        accs,
        best_results["se_accs"],
        sens,
        best_results["se_sens"],
        specs,
        best_results["se_specs"],
        precs,
        best_results["se_precs"],
        f1,
        best_results["se_f1"],
    )
)

# info.append('The parameters was selected from {}'.format(
#     'np.concatenate([np.logspace(-1.5,-1,20),np.logspace(-3,0,25), np.linspace(1.5,10,5), \
#     np.logspace(-2,1,30), np.logspace(-0.1, 1, 30), np.logspace(-0.8, 0,10)])')
#             )

# info.append('The parameters was selected from {}'.format(
#     '[0.5623, 0.0749, 0.2212, 1.7433, 0.0833, 0.9459]')
# )
# info.append('The parameters was selected from {}'.format(
#     'np.logspace(-2, 1, 30)')
#             )

info.append("The parameters was selected from {}".format("np.logspace(-2, 0, 30)"))
info.append("The leaf was selected from {}".format(args.leaf_list))
info.append("Random state:{}".format(rand))

filename = args.outputFile
dirname = os.path.dirname(filename)
if not os.path.exists(dirname):
    os.makedirs(dirname)
with open(filename, "w") as file_handler:
    for item in info:
        file_handler.write("{}\n".format(item))


# python KNN_cte.py -i  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv   -l 10 -n 1 -o   knn/1nn_bi2.txt
# python KNN_cte.py -i  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv   -l 10 -n 3 -o   knn/3nn_bi2.txt
# python KNN_cte.py -i  /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv   -l 10 -n 5 -o   knn/5nn_bi2.txt


# python KNN_cte.py -i /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -e "multiclass" -l 10 30 50 70 90  -n 1 -o   knn/1nn_mul.txt
# python KNN_cte.py -i /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -e "multiclass" -l 10 30 50 70 90  -n 3 -o   knn/3nn_mul.txt
# python KNN_cte.py -i /local/wujimeng/code_jimeng/data/invivo/lc50_processed_jim.csv  -e "multiclass" -l 10 30 50 70 90  -n 5 -o   knn/5nn_mul.txt

# python KNN_cte.py -i  data/invivo/invivo_repeated_w_invitro.csv -l 10 30 50  -n 1 3 5 -o   invitro/knn.txt
# python KNN_cte.py -i  data/invivo/invivo_repeated_w_invitro.csv   -e "multiclass" -l 10 30 50  -n 1 3 5 -o   invitro/knn_mul.txt

# python KNN_cte.py -i data/invitro/invitro_merged.csv -l 10 30 50 -n 1 3 5 -o knn/invitro_bi.txt
