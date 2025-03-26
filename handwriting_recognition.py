import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

# Load and preprocess dataset
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1)) / 255.0
test_images = test_images.reshape((10000, 28, 28, 1)) / 255.0

print("Train Images Shape:", train_images.shape)
print("Test Images Shape:", test_images.shape)

# Define the CNN model
model = models.Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Train the model
epochs = 5
history = model.fit(train_images, train_labels, epochs=epochs, validation_data=(test_images, test_labels))

# Plot training results
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Training Performance')
plt.show()

# Test single image
image = test_images[1].reshape(1, 28, 28, 1)
prediction = np.argmax(model.predict(image))
plt.imshow(image.reshape(28, 28), cmap='gray')
plt.title(f'Predicted Digit: {prediction}')
plt.show()

# Test multiple images
fig, axes = plt.subplots(1, 5, figsize=(12, 3))
for i in range(5):
    image = test_images[i].reshape(1, 28, 28, 1)
    prediction = np.argmax(model.predict(image))
    axes[i].imshow(image.reshape(28, 28), cmap='gray')
    axes[i].set_title(f'Pred: {prediction}')
    axes[i].axis('off')
plt.show()

# Save model
model.save("handwritten_digit_cnn.h5")

# Load and test saved model
loaded_model = keras.models.load_model("handwritten_digit_cnn.h5")
image = test_images[2].reshape(1, 28, 28, 1)
prediction = np.argmax(loaded_model.predict(image))
plt.imshow(image.reshape(28, 28), cmap='gray')
plt.title(f'Predicted Digit: {prediction}')
plt.show()
