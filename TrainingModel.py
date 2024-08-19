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


def load_dataset_from_csv(csv_file, meta, target_size=(224, 224)):
    images = []
    labels = []
    features = []
    with open(csv_file, mode='r', newline='') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row

        for row in reader:
            label = int(row[0])  # Read the label (convert it to int if necessary)
            bs_type = row[1]
            if bs_type not in ["2", "2s"]:
                continue
            file_path = row[2]  # Read the filepath

            feature = row[3]
            feature = {int(k): float(v) for k, v in (item.split(':') for item in feature.split())}

            missing = -9999999
            feature_arr = [missing] * len(meta)
            for feat_name, feat_value in features:
                if feat_name in meta:
                    feature_arr[meta[feat_name]] = feat_value
            feature_arr = [feature_arr]
            # Process the label and filepath as needed
            print(f"Label: {label}, Filepath: {file_path}")

            try:
                # 打开图片并调整大小
                img = Image.open(file_path).resize(target_size)
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                # 将图片转换为 NumPy 数组
                img_array = np.array(img)
                images.append(img_array)
                labels.append(label)
                features.append(feature)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

    return np.array(images, dtype=float), np.array(labels, dtype=float), np.array(features, dtype=float)


def train_model(code):
    if os.path.exists(f"./TMP/{code}_images.npy"):
        images = np.load(f"./TMP/{code}_images.npy")
        labels = np.load(f"./TMP/{code}_labels.npy")
        features = np.load(f"./TMP/{code}_features.npy")
    else:
        meta = json.load(open(f"./TMP/{code}_feature.meta", "r"))
        images, labels, features = load_dataset_from_csv(f"./TMP/{code}_dataset.csv",meta=meta, target_size=(224, 224))
        np.save(f"./TMP/{code}_images.npy", images)
        np.save(f"./TMP/{code}_labels.npy", labels)
        np.save(f"./TMP/{code}_features.npy", features)

    # images = keras.applications.resnet.preprocess_input(images)

    X_train, X_val, y_train, y_val, f_train, f_val = train_test_split(images, labels, features, test_size=0.2,
                                                                      shuffle=False,
                                                                      random_state=42)

    print(f"Training data: {X_train.shape}, Validation data: {X_val.shape}")

    # 模型构建
    conv_base = keras.applications.ResNet50(weights='imagenet', include_top=False,
                                            input_shape=(224, 224, 3))
    img_inputs = keras.layers.Input()
    output = conv_base(img_inputs)
    output = keras.layers.GlobalAveragePooling2D()(output)
    output = keras.layers.Dense(128, activation='relu')(output)
    output = keras.layers.Dropout(0.5)(output)
    output = keras.layers.Dense(16, activation='relu')(output)
    output = keras.layers.Dense(1, activation='sigmoid')(output)
    model = keras.models.Model(inputs=img_inputs, outputs=output)
    # 冻结卷积基
    conv_base.trainable = False

    # 编译模型
    model.compile(loss=keras.losses.BinaryCrossentropy(),
                  optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  metrics=[keras.metrics.AUC(name='auc')])

    class_weights = compute_class_weight(
        "balanced", classes=np.unique(labels), y=labels
    )
    class_weight = {
        0: class_weights[0],
        1: class_weights[1],
    }
    print(f"class_weight:{class_weight}")
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_auc',
        mode='max',
        patience=50,
        restore_best_weights=True,
        start_from_epoch=10
    )

    # 训练模型
    model.fit(
        X_train,
        y_train,
        class_weight=class_weight,
        epochs=100,
        verbose=2,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping])

    model.save(f"./TMP/{code}_model.keras")


if __name__ == "__main__":
    code = "EURUSD"
    train_model(code)
