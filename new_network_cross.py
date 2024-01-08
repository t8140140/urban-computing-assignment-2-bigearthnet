import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import TensorBoard
from datetime import datetime
from sklearn.model_selection import StratifiedKFold


folders_clean = np.load("arrays/clean_folders.npy")

input_clean = np.load("arrays/clean_input.npy")
input_clean = input_clean.reshape(len(input_clean),120,120,3)
input_shape = input_clean.shape
input_clean = tf.squeeze(input_clean)
labels_clean = np.load("arrays/clean_labels.npy")
labels_categorical = tf.keras.utils.to_categorical(labels_clean, num_classes=10)
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
model.add(layers.Dense(64))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.5))

# Output layer with sigmoid activation (for binary classification)
model.add(layers.Dense(10, activation='softmax'))

# Display the model summary
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Cross validation
num_folds = 5  # Adjust as needed

# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

# Loop over the folds
for fold, (train_index, val_index) in enumerate(skf.split(input_clean, labels_clean)):
    print(f"\nTraining on fold {fold + 1}/{num_folds}")

    # Create train and validation sets for this fold
    input_train, input_val = input_clean[train_index], input_clean[val_index]
    labels_train, labels_val = labels_categorical[train_index], labels_categorical[val_index]

    # Create a new timestamp and log directory for each fold
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/new_{timestamp}_fold{fold + 1}/"
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    checkpoint_path = f"weights/new_{timestamp}_fold{fold + 1}.h5"
    model_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, save_weights_only=True)

    # Fit the model on the current fold
    history = model.fit(
        input_train, labels_train,
        epochs=2, batch_size=32, verbose=1,
        validation_data=(input_val, labels_val),
        callbacks=[tensorboard_callback, model_callback]
    )