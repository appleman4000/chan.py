# cython: language_level=3
# encoding:utf-8

import os

from sklearn.utils import compute_class_weight

os.environ['KERAS_BACKEND'] = 'torch'
import csv

import keras
import numpy as np

from keras import Sequential
from PIL import Image
from sklearn.model_selection import train_test_split


def load_dataset_from_csv(csv_file, target_size=(224, 224)):
    images = []
    labels = []
    with open(csv_file, mode='r', newline='') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row

        for row in reader:
            label = int(row[0])  # Read the label (convert it to int if necessary)
            filepath = row[1]  # Read the filepath

            # Process the label and filepath as needed
            print(f"Label: {label}, Filepath: {filepath}")

            try:
                # 打开图片并调整大小
                img = Image.open(filepath).resize(target_size)
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                # 将图片转换为 NumPy 数组
                img_array = np.array(img)
                images.append(img_array)
                labels.append(label)
            except Exception as e:
                print(f"Error processing file {filepath}: {e}")

    return np.array(images, dtype=float), np.array(labels, dtype=float)


def train_model(code):
    if os.path.exists(f"./TMP/{code}_images.npy"):
        images = np.load(f"./TMP/{code}_images.npy")
        labels = np.load(f"./TMP/{code}_labels.npy")
    else:
        images, labels = load_dataset_from_csv(f"./TMP/{code}_dataset.csv", target_size=(224, 224))
        np.save(f"./TMP/{code}_images.npy", images)
        np.save(f"./TMP/{code}_labels.npy", labels)

    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, shuffle=False, random_state=42)

    print(f"Training data: {X_train.shape}, Validation data: {X_val.shape}")

    # 模型构建
    conv_base = keras.applications.ConvNeXtSmall(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model = Sequential()
    model.add(conv_base)
    model.add(keras.layers.GlobalAveragePooling2D())
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    # 冻结卷积基
    conv_base.trainable = False

    # 编译模型
    model.compile(loss=keras.losses.BinaryCrossentropy(),
                  optimizer=keras.optimizers.Adam(learning_rate=1e-3, weight_decay=0.004),
                  metrics=[keras.metrics.AUC(name='auc')])
    #
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
        patience=20,
        restore_best_weights=True
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
