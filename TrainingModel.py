# cython: language_level=3
# encoding:utf-8
import json
import os

from sklearn.utils import compute_class_weight

os.environ['KERAS_BACKEND'] = 'torch'
import csv

import keras

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
                img_array = np.array(img, np.float32)
                assert img_array.shape == target_size + (3,)
                images.append(img_array)
                labels.append(label)
                features.append(feature_arr)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.float32), np.array(features, dtype=np.float32)


import h5py
import numpy as np


def append_to_hdf5(file_name, dataset_name, new_data):
    """
    追加多维数组到指定的 HDF5 数据集中，并返回整个数据集内容。

    参数:
    - file_name: str, HDF5 文件名
    - dataset_name: str, 数据集名称
    - new_data: numpy.ndarray, 要追加的数据

    返回:
    - entire_data: numpy.ndarray, 包含所有数据的数据集内容
    """
    # 打开 HDF5 文件，如果文件不存在，则创建它
    with h5py.File(file_name, 'a') as hdf5_file:
        # 检查数据集是否存在
        if dataset_name in hdf5_file:
            # 如果数据集已存在，获取现有数据集
            dataset = hdf5_file[dataset_name]
        else:
            # 如果数据集不存在，创建新的数据集
            dataset = hdf5_file.create_dataset(
                dataset_name,
                shape=(0,) + new_data.shape[1:],  # 初始形状，第一维度为 0，其他维度与新数据一致
                maxshape=(None,) + new_data.shape[1:],  # 允许第一维度扩展
                dtype=new_data.dtype,
                chunks=True,  # 使用块存储以支持动态扩展
                compression='gzip',  # 启用 Gzip 压缩
                compression_opts=9  # 压缩级别，0-9 之间，9 为最高压缩级别
            )

        # 扩展数据集的第一维度以包含新数据
        dataset.resize(dataset.shape[0] + new_data.shape[0], axis=0)

        # 将新数据写入数据集
        dataset[-new_data.shape[0]:] = new_data

    return


# 示例用法
file_name = './TMP/mydata.h5'


def get_from_hdf5(file_name, dataset_name):
    with h5py.File(file_name, 'a') as hdf5_file:
        # 检查数据集是否存在
        if dataset_name in hdf5_file:
            # 如果数据集已存在，获取现有数据集
            dataset = hdf5_file[dataset_name]
        else:
            return None
        # 读取整个数据集内容并返回
        entire_data = dataset[:]

    return np.array(entire_data, dtype=np.float32)


def get_all_in_one_dataset(codes, bsp_type):
    if os.path.exists(file_name):
        X_train = get_from_hdf5(file_name, "X_train")
        X_val = get_from_hdf5(file_name, "X_val")
        y_train = get_from_hdf5(file_name, "y_train")
        y_val = get_from_hdf5(file_name, "y_val")
        f_train = get_from_hdf5(file_name, "f_train")
        f_val = get_from_hdf5(file_name, "f_val")
        return X_train, X_val, y_train, y_val, f_train, f_val

    for code in codes:
        meta = json.load(open(f"./TMP/{code}_feature.meta", "r"))
        images, labels, features = load_dataset_from_csv(f"./TMP/{code}_dataset.csv", bsp_type=bsp_type, meta=meta,
                                                         target_size=(224, 224))
        images /= 255.0
        X_train, X_val, y_train, y_val, f_train, f_val = train_test_split(images, labels, features, test_size=0.2,
                                                                          shuffle=False,
                                                                          random_state=42)
        # 追加新数据到数据集中
        append_to_hdf5(file_name, "X_train", X_train)
        append_to_hdf5(file_name, "X_val", X_val)
        append_to_hdf5(file_name, "y_train", y_train)
        append_to_hdf5(file_name, "y_val", y_val)
        append_to_hdf5(file_name, "f_train", f_train)
        append_to_hdf5(file_name, "f_val", f_val)

    X_train = get_from_hdf5(file_name, "X_train")
    X_val = get_from_hdf5(file_name, "X_val")
    y_train = get_from_hdf5(file_name, "y_train")
    y_val = get_from_hdf5(file_name, "y_val")
    f_train = get_from_hdf5(file_name, "f_train")
    f_val = get_from_hdf5(file_name, "f_val")

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
    conv_base = keras.applications.ConvNeXtSmall(weights='imagenet', include_top=False,
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

    model.compile(loss=keras.losses.BinaryCrossentropy(),
                  optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  metrics=[keras.metrics.AUC(name='auc')])
    # 训练模型
    model.fit((X_train, f_train), y_train, epochs=50, verbose=2, batch_size=32, class_weight=class_weight,
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
    # X_train, X_val, y_train, y_val, f_train, f_val = get_one_dataset("EURUSD", bsp_type=["2", "2s"])
    train_model(code="all_in_one", bsp_type=["2", "2s"], X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val,
                f_train=f_train, f_val=f_val)
