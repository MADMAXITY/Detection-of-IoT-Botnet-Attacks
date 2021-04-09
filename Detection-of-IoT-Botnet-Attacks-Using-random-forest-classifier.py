import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import (
    confusion_matrix,
    recall_score,
    classification_report,
    f1_score,
    accuracy_score,
)
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn import svm
import itertools


def dataPrepare():
    data = pd.read_csv("Data/benign_traffic.csv")
    data["Class"] = 0
    udp = pd.read_csv("Data/mirai_attacks/udp.csv")
    udp["Class"] = 1

    ack = pd.read_csv("Data/mirai_attacks/ack.csv")
    ack["Class"] = 1

    scan = pd.read_csv("Data/mirai_attacks/scan.csv")
    scan["Class"] = 1

    syn = pd.read_csv("Data/mirai_attacks/syn.csv")
    syn["Class"] = 1

    udpplain = pd.read_csv("Data/mirai_attacks/udpplain.csv")
    udpplain["Class"] = 1

    g_combo = pd.read_csv("Data/gafgyt_attacks/combo.csv")
    g_combo["Class"] = 1

    g_junk = pd.read_csv("Data/gafgyt_attacks/junk.csv")
    g_junk["Class"] = 1

    g_scan = pd.read_csv("Data/gafgyt_attacks/scan.csv")
    g_scan["Class"] = 1

    g_tcp = pd.read_csv("Data/gafgyt_attacks/tcp.csv")
    g_tcp["Class"] = 1

    g_udp = pd.read_csv("Data/gafgyt_attacks/udp.csv")
    g_udp["Class"] = 1

    frames = [
        data,
        udp,
        ack,
        scan,
        syn,
        udpplain,
        g_combo,
        g_junk,
        g_scan,
        g_tcp,
        g_udp,
    ]

    result = pd.concat(frames, ignore_index=True)
    return result


def dataPreprocessing(data):
    print("------")

    count_class = pd.value_counts(data["Class"], sort=True).sort_index()
    print(count_class)

    print("------")
    X = data.iloc[:, data.columns != "Class"]
    y = data.iloc[:, data.columns == "Class"]

    positive_sample_count = len(data[data.Class == 1])
    print(positive_sample_count)

    negative_sample_index = np.array(data[data.Class == 0].index)
    print(negative_sample_index[:5])

    positive_sample_index = data[data.Class == 1].index

    random_positive_sample_index = np.random.choice(
        positive_sample_index, int(1 * len(data[data.Class == 0])), replace=False
    )

    print(random_positive_sample_index[:5])

    under_sample_index = np.concatenate(
        [random_positive_sample_index, negative_sample_index]
    )
    under_sample_data = data.iloc[under_sample_index, :]
    X_under_sample = under_sample_data.iloc[:, under_sample_data.columns != "Class"]
    y_under_sample = under_sample_data.iloc[:, under_sample_data.columns == "Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )

    (
        X_train_under_sample,
        X_test_under_sample,
        y_train_under_sample,
        y_test_under_sample,
    ) = train_test_split(X_under_sample, y_under_sample, test_size=0.3, random_state=0)
    print(len(X_train_under_sample))
    print(len(X_test_under_sample))

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_under_sample,
        X_test_under_sample,
        y_train_under_sample,
        y_test_under_sample,
    )


result = dataPrepare()
(
    X_train,
    X_test,
    y_train,
    y_test,
    X_train_under_sample,
    X_test_under_sample,
    y_train_under_sample,
    y_test_under_sample,
) = dataPreprocessing(result)

best_c_param = 10

lr = RandomForestClassifier()
lr.fit(X_train_under_sample, y_train_under_sample.values.ravel())


y_undersample_pred = lr.predict(X_test_under_sample.values)
print(
    "Accuracy Score : ", accuracy_score(y_undersample_pred, y_test_under_sample.values)
)
print(confusion_matrix(y_undersample_pred, y_test_under_sample.values))
