import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import TensorBoard
from datetime import datetime


folders_clean = np.load("arrays/clean_folders.npy")

input_clean = np.load("arrays/clean_input.npy")
input_clean = input_clean.reshape(len(input_clean),120,120,3)
input_shape = input_clean.shape

labels_clean = np.load("arrays/clean_labels.npy")
labels_categorical = tf.keras.utils.to_categorical(labels_clean, num_classes=10)

# same structure network but not pre-trained (initialised): init
input_t = tf.keras.Input(shape=(120,120,3))
model_res = tf.keras.applications.resnet50.ResNet50(
    include_top=False,
    weights="imagenet",
    input_tensor=input_t,
    input_shape=(120,120,3),
    pooling=None,
    classes=10,
)

for layer in model_res.layers:
    layer.trainable = False

# pretrained model
model = tf.keras.Sequential()
model.add(model_res)
model.add(layers.Flatten())
model.add(layers.BatchNormalization())
model.add(layers.Dense(32, activation="relu"))
model.add(layers.Dropout(0.5))
model.add(layers.BatchNormalization())
model.add(layers.Dense(10, activation="softmax"))
model.summary()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"logs/freeze_{timestamp}/"
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

checkpoint_path = f"weights/freeze_{timestamp}.h5"
# default metric to compare whether it's "best" is loss
model_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, save_weights_only=True)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(input_clean, labels_categorical, epochs=2, batch_size=32, verbose=1, validation_split=0.2, callbacks=[tensorboard_callback,model_callback])
