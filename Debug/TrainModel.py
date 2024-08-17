# cython: language_level=3
# encoding:utf-8
import csv
import os

import keras
import numpy as np
from sklearn.utils import class_weight

os.environ['KERAS_BACKEND'] = 'torch'
from keras import Sequential
from keras.src.applications.convnext import ConvNeXtTiny
from keras.src.layers import Dense
from PIL import Image
from sklearn.model_selection import train_test_split


def load_dataset_from_csv(csv_file, target_size=(240, 130)):
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


if os.path.exists('./images.npy'):
    images = np.load('./images.npy')
    labels = np.load('./labels.npy')
else:
    images, labels = load_dataset_from_csv("./dataset.csv", target_size=(224, 224))
    np.save('./images.npy', images)
    np.save('./labels.npy', labels)

X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, shuffle=False, random_state=42)

print(f"Training data: {X_train.shape}, Validation data: {X_val.shape}")

# 模型构建
conv_base = ConvNeXtTiny(weights='imagenet', include_top=False, input_shape=(240, 130, 3))

model = Sequential()
model.add(conv_base)
model.add(keras.layers.GlobalAveragePooling2D())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 冻结卷积基
conv_base.trainable = False

# 编译模型
model.compile(loss=keras.losses.BinaryFocalCrossentropy(),
              optimizer=keras.optimizers.Adam(learning_rate=1e-3),
              metrics=['accuracy'])

class_weights = class_weight.compute_class_weight(
    "balanced", classes=np.unique(labels), y=labels
)
class_weight: {
    0: class_weights[0],
    1: class_weights[1],
}
# 训练模型
history = model.fit(
    X_train,
    y_train,
    class_weight=class_weight,
    epochs=30,
    verbose=2,
    batch_size=256,
    validation_data=(X_val, y_val))
model.save("./model.h5")
