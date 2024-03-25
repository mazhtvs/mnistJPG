import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist         # библиотека базы выборок Mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing import image

# Загрузка набора данных MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Стандартизация входных данных
x_train = x_train / 255
x_test = x_test / 255
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

# Определение модели
model = keras.Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Компиляция модели
model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])

# Обучение модели
model.fit(x_train, y_train_cat, batch_size=32, epochs=5, validation_split=0.2)

# Получение текущей рабочей директории
folder_path = os.getcwd()

for i in range(1, 5):
    img_path = os.path.join(folder_path, f"5_{i}.jpg")
    img = image.load_img(img_path, color_mode="grayscale", target_size=(28, 28))

    img_array = image.img_to_array(img)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Предсказание
    res = model.predict(img_array)
    predicted_number = np.argmax(res)
    print(f"Предсказанное число для изображения {img_path}: {predicted_number}")

    # Отображение изображения
    plt.imshow(img, cmap=plt.cm.binary)
    plt.show()