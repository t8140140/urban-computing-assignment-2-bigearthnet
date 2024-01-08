import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

from datetime import datetime


folders_clean = np.load("arrays/clean_folders.npy")

input_clean = np.load("arrays/clean_input.npy")
input_clean = input_clean.reshape(len(input_clean),120,120,3)
input_shape = input_clean.shape
input_clean = tf.squeeze(input_clean)
print(input_clean.shape)
labels_clean = np.load("arrays/clean_labels.npy")
labels_categorical = tf.keras.utils.to_categorical(labels_clean, num_classes=10)

test_split = True
if test_split:
    input_clean = input_clean.numpy()

    print(labels_categorical[:10])
    input_train, input_test, labels_train, labels_test = train_test_split(
        input_clean, labels_categorical, test_size=0.2, random_state=42
    )

    input_train = tf.convert_to_tensor(input_train, dtype=tf.float32)
    labels_train = tf.convert_to_tensor(labels_train, dtype=tf.float32)
    input_test = tf.convert_to_tensor(input_test, dtype=tf.float32)
    labels_test = tf.convert_to_tensor(labels_test, dtype=tf.float32)

# Define the CNN model
model = models.Sequential()

# Convolutional layer with batch normalization and ReLU activation
model.add(layers.Conv2D(32, (3, 3), input_shape=(120,120,3)))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))

# Max pooling layer
model.add(layers.MaxPooling2D((2, 2)))

# Convolutional layer with batch normalization, ReLU activation, and dropout
model.add(layers.Conv2D(64, (3, 3)))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.25))

# Max pooling layer
model.add(layers.MaxPooling2D((2, 2)))

# Flatten the output before the fully connected layers
model.add(layers.Flatten())

# Fully connected layer with batch normalization, ReLU activation, and dropout
model.add(layers.Dense(32))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.5))

# Output layer with sigmoid activation (for binary classification)
model.add(layers.Dense(10, activation='softmax'))

# Display the model summary
model.summary()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# Compile the model
log_dir = f"logs/new_{timestamp}/"  # Choose a suitable directory
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
# callback = tf.keras.callbacks.ModelCheckpoint(filepath='beNet50.h5', monitor='acc', mode="max", save_best_only=True)
checkpoint_path = f"weights/new_{timestamp}.h5"
# default metric to compare whether it's "best" is loss
model_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, save_weights_only=True)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(
    input_train, labels_train,
    epochs=2, batch_size=32, verbose=1,
    validation_split=0.2,
    callbacks=[tensorboard_callback, model_callback]
)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(input_test, labels_test, verbose=2)
print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")

predictions = model.predict(input_clean)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(labels_clean, axis=1)


precision = precision_score(true_labels, predicted_labels, average='weighted')
recall = recall_score(true_labels, predicted_labels, average='weighted')
f1 = f1_score(true_labels, predicted_labels, average='weighted')
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
