import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import numpy as np
from PIL import Image
import os

# === 1. Подготовка данных ===
data_dir = 'dataset'
img_size = (224, 224)
batch_size = 32

# Аугментация для обучающего набора
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

# Только нормализация для валидации
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

val_data = val_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# === 2. Создание модели ===
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# === 3. Обучение ===
model.fit(train_data, validation_data=val_data, epochs=5)

# === 4. Сохранение модели ===
model.save('hotdog_model.h5')

# === 5. Предсказание по новой картинке ===
def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"[❌] File not found: {image_path}")
        return

    img = Image.open(image_path).resize(img_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    probability = prediction * 100  # переводим в проценты
    label = "HOTDOG 🌭" if prediction > 0.5 else "NOT HOTDOG ❌"
    status = "✅ Suitable" if prediction > 0.5 else "⚠️ Not suitable"

    print(f"[{status}] {os.path.basename(image_path)} → {label} | Confidence: {probability:.2f}%")

# === 6. Предсказание по всем изображениям в папке ===
def predict_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, filename)
            predict_image(image_path)

# === 7. Пример использования ===
predict_folder('dataset_test')