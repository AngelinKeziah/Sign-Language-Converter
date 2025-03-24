import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ✅ Correct dataset path
data_dir = "D:/SAK THIS PC DOWNLOADS/SignLanguageConverter/asl proj/asl_alphabet_train/asl_alphabet_train"

# ✅ Use ImageDataGenerator to load images dynamically
data_gen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.2
)

# ✅ Larger batch size for faster training
batch_size = 64

# ✅ Load training data dynamically
train_gen = data_gen.flow_from_directory(
    data_dir,
    target_size=(48, 48),  # ✅ Smaller size for faster processing
    batch_size=batch_size,
    class_mode='sparse',
    subset='training'
)

# ✅ Load validation data dynamically
val_gen = data_gen.flow_from_directory(
    data_dir,
    target_size=(48, 48),
    batch_size=batch_size,
    class_mode='sparse',
    subset='validation'
)

# ✅ Get class labels
class_labels = list(train_gen.class_indices.keys())
print(f"Classes found: {class_labels}")

# ✅ Build the model
print("Building the model...")
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(class_labels), activation='softmax')
])

# ✅ Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("Model compiled successfully! 🚀")

# ✅ Start training with optimized parameters
print("Starting training... 🚦")
history = model.fit(
    train_gen,
    steps_per_epoch=len(train_gen) // 4,  # ✅ Faster iteration per epoch
    epochs=5,
    validation_data=val_gen,
    validation_steps=len(val_gen) // 4
)
print("Training completed successfully! 🎯")

# ✅ Save the model
model_path = "D:/SAK THIS PC DOWNLOADS/SignLanguageConverter/asl proj/models/model.h5"
model.save(model_path)
print(f"Model saved successfully at: {model_path} ✅")
