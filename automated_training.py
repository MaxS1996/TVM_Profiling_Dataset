import json
from operator import index
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle
import time
date_time = "results_"+str(int(time.time()))
os.mkdir(date_time)

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import ExtraTreesRegressor
import xgboost as xgb

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

import warnings
warnings.filterwarnings('ignore', '.*sliced data.*', )

def prepare_dataset(path : str, workload_name : str):
    dataset = {}
    files = os.listdir(path)

    for file in files:
        with open(path+"/"+file, "r") as f:
            raw_str = f.read()
        data = json.loads(raw_str)

        if "layers" in data.keys():
            del data["layers"]

        if "relay" in data.keys():
            del data["relay"]

        if "conv2d" in workload_name:
            data["kernel_0"] = data["kernel"][0]
            data["kernel_1"] = data["kernel"][1]
            del data["kernel"]
            data["dilation_0"] = data["dilation"][0]
            data["dilation_1"] = data["dilation"][1]
            del data["dilation"]

            if not "kernel layout" in data.keys():
                data["kernel layout"] = "OIHW"

        elif workload_name == "dense":
            data["features"] = data["input shape"][1]
            del data["input shape"]
            del data["output shape"]
            #print()

        elif workload_name in ["max_pool2d", "avg_pool2d"]:
            data["pool_0"] = data["pool_size"][0]
            data["pool_1"] = data["pool_size"][1]
            del data["pool_size"]

        if workload_name in ["max_pool2d", "avg_pool2d", "conv2d", "dilated_conv2d", "depthwise_conv2d"]:
            del data["padding"]
            data["C_I"] = data["input shape"][3]
            data["H_I"] = data["input shape"][1]
            data["W_I"] = data["input shape"][2]
            del data["input shape"]

            data["C_O"] = data["output shape"][3]
            data["H_O"] = data["output shape"][1]
            data["W_O"] = data["output shape"][2]
            del data["output shape"]

            key = "strides"
            if "stride" in data.keys():
                key = "stride"
            #print(key)
            data["strides_0"] = data[key][0]
            data["strides_1"] = data[key][1]
            del data["strides"]
            if "stride" in data.keys():
                del data["stride"]

        dataset[file] = data

    return dataset

def create_dataframe(dataset : dict, workload_name : str):
    df = pd.DataFrame.from_dict(dataset, orient='index')
    categoricals = [
        "output dtype",
        "compute dtype",
        "workload",
        ]
    if workload_name in ["conv2d", "max_pool2d", "avg_pool2d", "depthwise_conv2d", "dilated_conv2d"]:
        categoricals += [
            #"padding",
            "data layout",
        ]
    if "conv2d" in workload_name:
        categoricals += [
            "kernel layout",
        ]

    for col in categoricals:
        oh = pd.get_dummies(df[col], prefix=col, drop_first=False)
        df = pd.concat([df, oh], axis=1).drop(col, axis=1)
    
    features = list(df.columns)
    labels = ["time", "power", "memory"]
    for label in labels:
        idx = features.index(label)
        del features[idx]
    del idx
    df = df.drop_duplicates(subset=features)

    output = pd.concat([df["time"], df["power"], df["memory"]], axis=1)
    df.pop("time")
    df.pop("power")
    df.pop("memory")

    return df, output


dataset_base = "./dataset"
targets = os.listdir(dataset_base)


layer_targets = list()
for target in targets:
    target_path = dataset_base + "/" + target
    layers = os.listdir(target_path)
    
    for layer in layers:
        dataset_path = target_path + "/" + layer + "/"
        print(dataset_path)
        layer_targets.append(dataset_path)

print("found {0} folders with samples, going to train models for each of these targets".format(len(layer_targets)))
print()


results = {}
for target in layer_targets:
    tmp = target.split("/")
    workload_name = tmp[-2]
    device_name = tmp[-3]
    files = os.listdir(target)
    dataset = {}

    print("{} : {}\t:\t contains {} samples".format(workload_name, device_name, len(files)))
    dataset = prepare_dataset(target, workload_name)
    print("\tLoading data into memory:\tcompleted")
    df, output = create_dataframe(dataset, workload_name)
    print("\tCreating dataframe:\t\tcompleted")
    print("\t[INFO] Remaining Samples after duplicate elimination:\t{}".format(len(df)))
    print("\t[INFO] Features that are used as predictor inputs   :\n\t\t{}".format(list(df.columns)))
    print("\t[INFO] Metrics that are going to be predicted       :\n\t\t{}".format(list(output.columns)))
    print()
    X = df.to_numpy()
    Y = output.to_numpy()
    labels = list(output.columns)

    if not os.path.exists(date_time+"/"+device_name):
        os.mkdir(date_time+"/"+device_name)
    os.mkdir(date_time+"/"+device_name+"/"+workload_name)

    with open(date_time+"/"+device_name+"/"+workload_name+"/features.json", "w") as f:
        f.write(json.dumps(list(df.columns)))

    X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2)
    for idx, name in enumerate(labels):
        model = xgb.XGBRegressor()
        #model = ExtraTreesRegressor()
        model.fit(X_train, y_train[:,idx])

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        r2_train = r2_score(y_train[:,idx], y_train_pred)
        r2_test = r2_score(y_test[:,idx], y_test_pred)

        mae_train = mean_absolute_error(y_train[:,idx], y_train_pred)
        mae_test = mean_absolute_error(y_test[:,idx], y_test_pred)

        mape_train = mean_absolute_percentage_error(y_train[:,idx], y_train_pred)
        mape_test = mean_absolute_percentage_error(y_test[:,idx], y_test_pred)

        print(name)
        print("\tR2    (train|test):\t{:.5f}\t\t{:.5f}".format(r2_train, r2_test))
        print("\tMAE   (train|test):\t{:.5f}\t\t{:.5f}".format(mae_train, mae_test))
        print("\tMAPE  (train|test):\t{:.5f}\t\t{:.5f}".format(mape_train, mape_test))
        print()
        result = {
            "device" : device_name,
            "workload" : workload_name,
            "metric" : name,
            "predictor" : "xgb",
            "training set size": len(X_train),
            "validation set size": len(X_test),
            "r2_train" : r2_train,
            "r2_test" : r2_test,
            "mae_train" : mae_train,
            "mae_test" : mae_test,
            "mape_train" : mape_train,
            "mape_test" : mape_test,
            "minimum" : Y[:,idx].min(),
            "maximum" : Y[:,idx].max(),
            "mean" : Y[:,idx].mean(),
            "median" : np.median(Y[:,idx]),
        }
        results[device_name+"-"+workload_name+"-"+name] = result
        with open(date_time+"/"+device_name+"/"+workload_name+"/"+name+"_predictor.pkl", "wb") as f:
            pickle.dump(model, f)

results = pd.DataFrame.from_dict(results, "index")
results.to_csv(date_time+"/predictor_results.csv")
results.to_excel(date_time+"/predictor_results.xlsx")
results.to_html(date_time+"/predictor_results.html")
results.to_markdown(date_time+"/predictor_results.md")
print("done")