# cython: language_level=3
# encoding:utf-8
import json
import os

from sklearn.utils import compute_class_weight

os.environ['KERAS_BACKEND'] = 'torch'
import csv

import keras
import numpy as np

from PIL import Image
from sklearn.model_selection import train_test_split


def load_dataset_from_csv(csv_file, meta, bsp_type, target_size=(224, 224)):
    images = []
    labels = []
    features = []
    with open(csv_file, mode='r', newline='') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row

        for row in reader:
            label = int(row[0])  # Read the label (convert it to int if necessary)
            bs_type = row[1]
            if bs_type not in bsp_type:
                continue
            file_path = row[2]  # Read the filepath

            feature = row[3]
            feature = {int(k): float(v) for k, v in (item.split(':') for item in feature.split())}

            missing = 0
            feature_arr = [missing] * len(meta)
            for feat_name, feat_value in feature.items():
                if feat_name in meta.values():
                    feature_arr[feat_name] = feat_value
            # Process the label and filepath as needed
            print(f"Label: {label}, Filepath: {file_path}")

            try:
                # 打开图片并调整大小
                img = Image.open(file_path).resize(target_size)
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                # 将图片转换为 NumPy 数组
                img_array = np.array(img)
                assert img_array.shape == target_size + (3,)
                images.append(img_array)
                labels.append(label)
                features.append(feature_arr)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

    return np.array(images, dtype=float), np.array(labels, dtype=float), np.array(features, dtype=float)


def get_all_in_one_dataset(codes, bsp_type):
    X_trains, X_vals, y_trains, y_vals, f_trains, f_vals = [], [], [], [], [], []
    for code in codes:
        meta = json.load(open(f"./TMP/{code}_feature.meta", "r"))
        images, labels, features = load_dataset_from_csv(f"./TMP/{code}_dataset.csv", bsp_type=bsp_type, meta=meta,
                                                         target_size=(224, 224))
        images /= 255.0
        X_train, X_val, y_train, y_val, f_train, f_val = train_test_split(images, labels, features, test_size=0.2,
                                                                          shuffle=False,
                                                                          random_state=42)
        X_trains.extend(X_train)
        X_vals.extend(X_val)
        y_trains.extend(y_train)
        y_vals.extend(y_val)
        f_trains.extend(f_train)
        f_vals.extend(f_val)
    X_train = np.array(X_trains).astype(np.float32)
    X_val = np.array(X_vals).astype(np.float32)
    y_train = np.array(y_trains).astype(np.float32)
    y_val = np.array(y_vals).astype(np.float32)
    f_train = np.array(f_trains).astype(np.float32)
    f_val = np.array(f_vals).astype(np.float32)
    return X_train, X_val, y_train, y_val, f_train, f_val


def get_one_dataset(code, bsp_type):
    meta = json.load(open(f"./TMP/{code}_feature.meta", "r"))

    images, labels, features = load_dataset_from_csv(f"./TMP/{code}_dataset.csv", bsp_type=bsp_type, meta=meta,
                                                     target_size=(224, 224))
    images /= 255.0
    X_train, X_val, y_train, y_val, f_train, f_val = train_test_split(images, labels, features, test_size=0.2,
                                                                      shuffle=False,
                                                                      random_state=42)
    return X_train, X_val, y_train, y_val, f_train, f_val


def create_alexnet(input_shape=(224, 224, 3)):
    model = keras.Sequential()

    # 第一层卷积
    model.add(keras.layers.Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    # 第二层卷积
    model.add(keras.layers.Conv2D(256, (5, 5), padding='same', activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    # 第三层卷积
    model.add(keras.layers.Conv2D(384, (3, 3), padding='same', activation='relu'))

    # 第四层卷积
    model.add(keras.layers.Conv2D(384, (3, 3), padding='same', activation='relu'))

    # 第五层卷积
    model.add(keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    return model


def train_model(code, bsp_type, X_train, X_val, y_train, y_val, f_train, f_val):
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
        patience=10,
        restore_best_weights=True,
        verbose=2
    )
    # 模型构建
    conv_base = keras.applications.ConvNeXtTiny(weights='imagenet', include_top=False,
                                                input_shape=(224, 224, 3))
    img_inputs = keras.layers.Input(shape=(224, 224, 3))
    feature_inputs = keras.layers.Input(shape=(len(meta),))
    img_output = conv_base(img_inputs)
    img_output = keras.layers.GlobalAvgPool2D()(img_output)
    img_output = keras.layers.Dense(128, activation='relu')(img_output)
    img_output = keras.layers.Dropout(0.5)(img_output)

    # feature_output = keras.layers.Dense(64, activation='relu')(feature_inputs)
    # output = keras.layers.Concatenate()([img_output, feature_output])
    output = keras.layers.Dense(1, activation='sigmoid')(img_output)
    model = keras.models.Model(inputs=[img_inputs, feature_inputs], outputs=output)

    # 冻结卷积基

    conv_base.trainable = True
    # for layer in conv_base.layers[-10:]:
    #     layer.trainable = True

    model.compile(loss=keras.losses.BinaryFocalCrossentropy(),
                  optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  metrics=[keras.metrics.AUC(name='auc')])
    # 训练模型
    model.fit((X_train, f_train), y_train, epochs=50, verbose=2, batch_size=32,
              validation_data=((X_val, f_val), y_val),
              callbacks=[early_stopping])

    # model.compile(loss=keras.losses.BinaryCrossentropy(),
    #               optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    #               metrics=[keras.metrics.AUC(name='auc')])
    #
    # conv_base.trainable = True
    #
    # # 训练模型
    # model.fit((X_train, f_train), y_train, class_weight=class_weight, epochs=20, verbose=2, batch_size=32,
    #           validation_data=((X_val, f_val), y_val),
    #           callbacks=[early_stopping])

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
        "XAUUSD",
        "XAGUSD",
    ]
    X_train, X_val, y_train, y_val, f_train, f_val = get_all_in_one_dataset(symbols, bsp_type=["2", "2s"])
    train_model(code="all_in_one", bsp_type=["2", "2s"], X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val,
                f_train=f_train, f_val=f_val)
