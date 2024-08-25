# cython: language_level=3
# encoding:utf-8
import json
import os

import numpy as np
from sklearn.utils import compute_class_weight

os.environ['KERAS_BACKEND'] = 'torch'
import csv

import keras

from sklearn.model_selection import train_test_split


def load_dataset_from_csv(csv_file, meta, bsp_type):
    labels = []
    features = []
    with open(csv_file, mode='r', newline='') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row

        for row in reader:
            label = int(row[0])  # Read the label (convert it to int if necessary)
            bs_types = row[1]
            if np.array([t not in bs_types for t in bsp_type]).all():
                continue
            file_path = row[2]  # Read the filepath

            feature = row[3]
            feature = {int(k): float(v) for k, v in (item.split(':') for item in feature.split())}

            missing = 0  # float('nan')
            feature_arr = [missing] * len(meta)
            for feat_name, feat_value in feature.items():
                if feat_name in meta.values():
                    feature_arr[feat_name] = feat_value
            # Process the label and filepath as needed
            print(f"Label: {label}, Filepath: {file_path}")
            labels.append(label)
            features.append(feature_arr)

    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.float32),


def get_all_in_one_dataset(codes, bsp_type):
    X_trains, X_vals, y_trains, y_vals = [], [], [], []

    for code in codes:
        meta = json.load(open(f"./TMP/{code}_feature.meta", "r"))
        features, labels = load_dataset_from_csv(f"./TMP/{code}_dataset.csv", bsp_type=bsp_type, meta=meta)
        X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.1, shuffle=False,
                                                          random_state=42)
        X_trains.extend(X_train)
        X_vals.extend(X_val)
        y_trains.extend(y_train)
        y_vals.extend(y_val)
        # 追加新数据到数据集中

    return np.array(X_trains, dtype=np.float32), np.array(X_vals, dtype=np.float32), \
        np.array(y_trains, dtype=np.float32), np.array(y_vals, dtype=np.float32)


def train_model(code, bsp_type, X_train, X_val, y_train, y_val):
    if code == "all_in_one":
        meta = json.load(open(f"./TMP/EURUSD_feature.meta", "r"))
    else:
        meta = json.load(open(f"./TMP/{code}_feature.meta", "r"))
    print(f"Training data: {X_train.shape}, Validation data: {X_val.shape}")
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )
    class_weight = {
        0: class_weights[0],
        1: class_weights[1],
    }
    print(f"class_weight:{class_weight}")
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_auc',
        mode='max',
        patience=20,
        restore_best_weights=True,
        verbose=2
    )
    # 模型构建
    feature_inputs = keras.layers.Input(shape=(len(meta),))
    # 对输入因子数据进行归一化
    normalization_layer = keras.layers.Normalization(axis=-1)
    normalization_layer.adapt(X_train)
    feature_output = normalization_layer(feature_inputs)
    feature_output = keras.layers.Dense(64)(feature_output)
    feature_output = keras.layers.Activation("relu")(feature_output)
    output = keras.layers.Dense(1, activation='sigmoid')(feature_output)
    model = keras.models.Model(inputs=feature_inputs, outputs=output)

    model.compile(loss=keras.losses.BinaryCrossentropy(),
                  optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  metrics=[keras.metrics.AUC(name='auc')])
    # 训练模型
    model.fit(X_train, y_train, epochs=50, verbose=2, batch_size=32, class_weight=class_weight,
              validation_data=(X_val, y_val),
              callbacks=[early_stopping])

    model.save(f"./TMP/{code}_{'_'.join(bsp_type)}_model.keras")


if __name__ == "__main__":
    symbols = [
        # Major
        "EURUSD",
        "GBPUSD",
        "AUDUSD",
        "NZDUSD",
        "USDJPY",
        "USDCAD",
        "USDCHF",
        # Crosses
        "AUDCHF",
        "AUDJPY",
        "AUDNZD",
        "CADCHF",
        "CADJPY",
        "CHFJPY",
        "EURAUD",
        "EURCAD",
        "AUDCAD",
        "EURCHF",
        "GBPNZD",
        "GBPCAD",
        "GBPCHF",
        "GBPJPY",
    ]
    # X_train, X_val, y_train, y_val = get_one_dataset("EURUSD", bsp_type=["2", "2s"])
    X_train, X_val, y_train, y_val = get_all_in_one_dataset(symbols, bsp_type=["1", "1p"])
    train_model(code="all_in_one", bsp_type=["1", "1p"], X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val)
