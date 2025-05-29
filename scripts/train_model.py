import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import os

# Directorio del dataset
DATASET_DIR = 'dataset/'

# Parámetros
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_INITIAL = 25
EPOCHS_FINE_TUNE = 10
NUM_CLASSES = 5

# Generador de imágenes con aumentación
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    rotation_range=15,
    shear_range=0.1,
    fill_mode='nearest'
)

train_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Cargar MobileNet preentrenado sin la parte superior
base_model = tf.keras.applications.MobileNet(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

# Agregar capas finales personalizadas
x = base_model.output
x = GlobalAveragePooling2D()(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

# Compilar modelo (entrenamiento inicial)
model.compile(optimizer=Adam(learning_rate=1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entrenamiento inicial
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_INITIAL,
    callbacks=[early_stop]
)

# Guardar modelo inicial
model.save('model/model_initial.h5')
print("✅ Modelo inicial guardado en model/model_initial.h5")

# 🔓 Fine-tuning: descongelar base_model
base_model.trainable = True

# Recompilar con menor tasa de aprendizaje
model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entrenamiento de ajuste fino
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_FINE_TUNE,
    callbacks=[early_stop]
)

# Guardar modelo final ajustado
model.save('model/model.h5')
print("✅ Modelo final ajustado guardado en model/model.h5")
