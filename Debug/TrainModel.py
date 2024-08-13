# encoding:utf-8
import os

import keras
import numpy as np

os.environ['KERAS_BACKEND'] = 'torch'
from keras import Sequential
from keras.src.applications.convnext import ConvNeXtTiny
from keras.src.layers import Dense
from PIL import Image
from sklearn.model_selection import train_test_split


#


# # 数据增强
# datagen = ImageDataGenerator(
#     rescale=1. / 255,
#     # rotation_range=20,
#     # width_shift_range=0.2,
#     # height_shift_range=0.2,
#     # shear_range=0.2,
#     # zoom_range=0.2,
#     # horizontal_flip=True,
#     # fill_mode='nearest',
#     validation_split=0.2,
# )
# train_generator = datagen.flow_from_directory(
#     target_train_dir,
#     target_size=(240, 130),
#     batch_size=256,
#     class_mode='binary',
#     subset='training')
#
# validation_generator = datagen.flow_from_directory(
#     target_train_dir,
#     target_size=(240, 130),
#     batch_size=256,
#     class_mode='binary',
#     subset='validation')
def load_images_from_directory(directory, target_size=(240, 130)):
    images = []
    labels = []
    filenames = []

    for label in os.listdir(directory):
        label_dir = os.path.join(directory, label)
        if os.path.isdir(label_dir):
            for filename in os.listdir(label_dir):
                file_path = os.path.join(label_dir, filename)
                try:
                    # 打开图片并调整大小
                    img = Image.open(file_path).resize(target_size)
                    if img.mode == 'RGBA':
                        img = img.convert('RGB')
                    # 将图片转换为 NumPy 数组
                    img_array = np.array(img)
                    images.append(img_array)
                    labels.append(label)
                    filenames.append(filename)
                    print(f"{file_path} {label}")
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

    return np.array(images, dtype=float), np.array(labels, dtype=float), np.array(filenames, dtype=str)


if os.path.exists('./images.npy'):
    images = np.load('./images.npy')
    labels = np.load('./labels.npy')
    filenames = np.load('./filenames.npy')
else:
    target_train_dir = './PNG/TRAIN'
    images, labels, filenames = load_images_from_directory(target_train_dir)
    np.save('./images.npy', images)
    np.save('./labels.npy', labels)
    np.save('./filenames.npy', filenames)

sorted_indices = np.argsort(filenames[:, 6:-4])
images = images[sorted_indices]
labels = labels[sorted_indices]
filenames = filenames[sorted_indices]
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, shuffle=False, random_state=42,
                                                  stratify=labels)

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
# 获取类别及其样本数
# class_names = os.listdir(target_train_dir)
# class_names.sort()  # 确保类别顺序一致
# num_classes = len(class_names)
#
# # 统计每个类别的样本数
# class_counts = [len(os.listdir(os.path.join(target_train_dir, class_name))) for class_name in class_names]
#
# # 计算类别权重
# class_weights = compute_class_weight(
#     class_weight='balanced',
#     classes=np.arange(num_classes),
#     y=np.concatenate([np.full(count, i) for i, count in enumerate(class_counts)])
# )
#
# class_weight_dict = dict(enumerate(class_weights))
# 训练模型
history = model.fit(
    X_train,
    y_train,
    epochs=30,
    verbose=2,
    batch_size=256,
    validation_data=(X_val, y_val))
model.save("./model.h5")
