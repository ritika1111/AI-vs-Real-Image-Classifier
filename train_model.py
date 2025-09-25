import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import (InputLayer, Conv2D, MaxPool2D, GlobalAveragePooling2D, 
                          Dense, ELU)
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Nadam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import keras

# === LOAD DATASETS ===
ai_dataset = keras.utils.image_dataset_from_directory(
    "dataset/AiArtData",
    image_size=(224, 224),
    batch_size=32,
    label_mode='int'
)

real_dataset = keras.utils.image_dataset_from_directory(
    "dataset/RealArt",
    image_size=(224, 224),
    batch_size=32,
    label_mode='int'
)

# === NORMALIZATION & LABEL FIXING ===
normalization_layer = tf.keras.layers.Rescaling(1./255)
normalized_ai_dataset = ai_dataset.map(lambda x, y: (normalization_layer(x), tf.ones_like(y)))
normalized_real_dataset = real_dataset.map(lambda x, y: (normalization_layer(x), tf.zeros_like(y)))

# === COMBINE DATA ===
combined_dataset = normalized_ai_dataset.concatenate(normalized_real_dataset)

images_list = []
labels_list = []

for images, labels in combined_dataset:
    for image, label in zip(images, labels):
        images_list.append(image.numpy())
        labels_list.append(label.numpy())

images_array = np.asarray(images_list)
labels_array = np.asarray(labels_list)

# === TRAIN-VAL-TEST SPLIT ===
x_train, x_temp, y_train, y_temp = train_test_split(images_array, labels_array, test_size=0.3, random_state=42)
x_test, x_val, y_test, y_val = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

# === MODEL ===
model = Sequential()
model.add(InputLayer(input_shape=(224, 224, 3)))


model.add(Conv2D(32, 3, activation=ELU(), padding="same"))
model.add(MaxPool2D())

model.add(Conv2D(64, 3, activation=ELU(), padding="same"))
model.add(MaxPool2D())

model.add(Conv2D(128, 3, activation=ELU(), padding="same"))
model.add(MaxPool2D())

model.add(Conv2D(256, 3, activation=ELU(), padding="same"))
model.add(MaxPool2D())

model.add(GlobalAveragePooling2D())
model.add(Dense(1024, activation=ELU()))
model.add(Dense(512, activation=ELU()))
model.add(Dense(256, activation=ELU()))
model.add(Dense(128, activation=ELU()))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Nadam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# === CALLBACKS ===
checkpoint = ModelCheckpoint('model.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, min_lr=1e-7)

# === TRAINING ===
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    batch_size=32,
    epochs=20,
    callbacks=[checkpoint, lr_scheduler],
    verbose=1
)

# === EVALUATION ===
model2 = keras.models.load_model('model.keras', custom_objects={"ELU": keras.layers.ELU})
y_pred = model2.predict(x_test)
y_pred = (y_pred > 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# === PLOTTING ===
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss per Epoch")
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy per Epoch")
plt.legend()
plt.show()
