import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Add, UpSampling2D
from tensorflow_addons.metrics import F1Score

from PIL import Image
from keras.callbacks import ModelCheckpoint
from keras_cv_attention_models.attention_layers import (
    activation_by_name, ChannelAffine, conv2d_no_bias, depthwise_conv2d_no_bias,
    drop_block, mlp_block, output_block, add_pre_post_process
)
from keras_cv_attention_models.download_and_load import reload_model_weights

# Configurable parameters
# Base directory placeholder
BASE_DIR = "path/to/base/directory"

config = {
    "TRAIN_IMAGES_DIR": os.path.join(BASE_DIR, "train/images"),
    "TRAIN_LABELS_DIR": os.path.join(BASE_DIR, "train_labels"),
    "TEST_IMAGES_DIR": os.path.join(BASE_DIR, "test/images"),
    "TEST_LABELS_DIR": os.path.join(BASE_DIR, "test_labels"),
    "DATASET_FILE": os.path.join(BASE_DIR, "Dataset_train.h5"),
    "IMAGE_SIZE": (384, 384),
    "EPOCHS": 10,
    "BATCH_SIZE": 4,
    "CHECKPOINT_FILE": os.path.join(BASE_DIR, "unet_membrane.hdf5"),
    "MODEL_SUMMARY_FILE": os.path.join(BASE_DIR, "model_summary.txt")
}

# Function to load and resize images
def load_and_resize_images(directory, image_size):
    images = []
    for index, image_file in enumerate(os.listdir(directory)):
        print(f"Image number: {index}")
        img = Image.open(os.path.join(directory, image_file))
        img = img.resize(image_size)
        arr = np.array(img)
        images.append(arr)
    return np.array(images)

# Load and preprocess training data
TrainX = load_and_resize_images(config["TRAIN_IMAGES_DIR"], config["IMAGE_SIZE"])
TrainY = load_and_resize_images(config["TRAIN_LABELS_DIR"], config["IMAGE_SIZE"])

# Load and preprocess test data
TestX = load_and_resize_images(config["TEST_IMAGES_DIR"], config["IMAGE_SIZE"])
TestY = load_and_resize_images(config["TEST_LABELS_DIR"], config["IMAGE_SIZE"])

# Reshape label arrays
TrainY = TrainY.reshape(TrainY.shape[0], TrainY.shape[1], TrainY.shape[2], 1)
TestY = TestY.reshape(TestY.shape[0], TestY.shape[1], TestY.shape[2], 1)

# Save dataset to HDF5 file
with h5py.File(config["DATASET_FILE"], 'w') as hdf:
    hdf.create_dataset('images', data=TrainX, compression='gzip', compression_opts=9)
    hdf.create_dataset('masks', data=TrainY, compression='gzip', compression_opts=9)

# Define custom layers and blocks
class MultiHeadRelativePositionalKernelBias(tf.keras.layers.Layer):
    def __init__(self, input_height=-1, is_heads_first=False, **kwargs):
        super().__init__(**kwargs)
        self.input_height = input_height
        self.is_heads_first = is_heads_first

    def build(self, input_shape):
        blocks, num_heads = (input_shape[2], input_shape[1]) if self.is_heads_first else (input_shape[1], input_shape[2])
        size = int(tf.math.sqrt(float(input_shape[-1])))
        height = self.input_height if self.input_height > 0 else int(tf.math.sqrt(float(blocks)))
        width = blocks // height
        pos_size = 2 * size - 1
        initializer = tf.initializers.truncated_normal(stddev=0.02)
        self.pos_bias = self.add_weight(name="positional_embedding", shape=(num_heads, pos_size * pos_size), initializer=initializer, trainable=True)

        idx_hh, idx_ww = tf.range(0, size), tf.range(0, size)
        coords = tf.reshape(tf.expand_dims(idx_hh, -1) * pos_size + idx_ww, [-1])
        bias_hh = tf.concat([idx_hh[: size // 2], tf.repeat(idx_hh[size // 2], height - size + 1), idx_hh[size // 2 + 1:]], axis=-1)
        bias_ww = tf.concat([idx_ww[: size // 2], tf.repeat(idx_ww[size // 2], width - size + 1), idx_ww[size // 2 + 1:]], axis=-1)
        bias_hw = tf.expand_dims(bias_hh, -1) * pos_size + bias_ww
        bias_coords = tf.expand_dims(bias_hw, -1) + coords
        bias_coords = tf.reshape(bias_coords, [-1, size**2])[::-1]

        bias_coords_shape = [bias_coords.shape[0]] + [1] * (len(input_shape) - 4) + [bias_coords.shape[1]]
        self.bias_coords = tf.reshape(bias_coords, bias_coords_shape)
        if not self.is_heads_first:
            self.transpose_perm = [1, 0] + list(range(2, len(input_shape) - 1))

    def call(self, inputs):
        if self.is_heads_first:
            return inputs + tf.gather(self.pos_bias, self.bias_coords, axis=-1)
        else:
            return inputs + tf.transpose(tf.gather(self.pos_bias, self.bias_coords, axis=-1), self.transpose_perm)

    def get_config(self):
        base_config = super().get_config()
        base_config.update({"input_height": self.input_height, "is_heads_first": self.is_heads_first})
        return base_config

def LWA(inputs, kernel_size=7, num_heads=4, key_dim=0, out_weight=True, qkv_bias=True, out_bias=True, attn_dropout=0, output_dropout=0, name=None):
    _, hh, ww, cc = inputs.shape
    key_dim = key_dim if key_dim > 0 else cc // num_heads
    qk_scale = 1.0 / (float(key_dim) ** 0.5)
    out_shape = cc
    qkv_out = num_heads * key_dim

    should_pad_hh, should_pad_ww = max(0, kernel_size - hh), max(0, kernel_size - ww)
    if should_pad_hh or should_pad_ww:
        inputs = tf.pad(inputs, [[0, 0], [0, should_pad_hh], [0, should_pad_ww], [0, 0]])
        _, hh, ww, cc = inputs.shape

    qkv = keras.layers.Dense(qkv_out * 3, use_bias=qkv_bias, name=name and name + "qkv")(inputs)
    query, key_value = tf.split(qkv, [qkv_out, qkv_out * 2], axis=-1)
    query = tf.expand_dims(tf.reshape(query, [-1, hh * ww, num_heads, key_dim]), -2)

    key_value = CompatibleExtractPatches(sizes=kernel_size, strides=1, padding="VALID", compressed=False)(key_value)
    padded = (kernel_size - 1) // 2
    key_value = tf.concat([tf.repeat(key_value[:, :1], padded, axis=1), key_value, tf.repeat(key_value[:, -1:], padded, axis=1)], axis=1)
    key_value = tf.concat([tf.repeat(key_value[:, :, :1], padded, axis=2), key_value, tf.repeat(key_value[:, :, -1:], padded, axis=2)], axis=2)

    key_value = tf.reshape(key_value, [-1, kernel_size * kernel_size, key_value.shape[-1]])
    key, value = tf.split(key_value, 2, axis=-1)
    key = tf.transpose(tf.reshape(key, [-1, key.shape[1], num_heads, key_dim]), [0, 2, 3, 1])
    key = tf.reshape(key, [-1, hh * ww, num_heads, key_dim, kernel_size * kernel_size])
    value = tf.transpose(tf.reshape(value, [-1, value.shape[1], num_heads, key_dim]), [0, 2, 1, 3])
    value = tf.reshape(value, [-1, hh * ww, num_heads, kernel_size * kernel_size, key_dim])

    attention_scores = keras.layers.Lambda(lambda xx: tf.matmul(xx[0], xx[1]))([query, key]) * qk_scale
    attention_scores = MultiHeadRelativePositionalKernelBias(input_height=hh, name=name and name + "pos")(attention_scores)
    attention_scores = keras.layers.Softmax(axis=-1, name=name and name + "attention_scores")(attention_scores)
    attention_scores = keras.layers.Dropout(attn_dropout, name=name and name + "attn_drop")(attention_scores) if attn_dropout > 0 else attention_scores

    attention_output = keras.layers.Lambda(lambda xx: tf.matmul(xx[0], xx[1]))([attention_scores, value])
    attention_output = tf.reshape(attention_output, [-1, hh, ww, num_heads * key_dim])

    if should_pad_hh or should_pad_ww:
        attention_output = attention_output
        if should_pad_hh or should_pad_ww:
            attention_output = attention_output[:hh, :ww, :]

    if out_weight:
        attention_output = keras.layers.Conv2D(out_shape, kernel_size=1, use_bias=out_bias, name=name and name + "out")(attention_output)
    else:
        attention_output = keras.layers.Dense(out_shape, use_bias=out_bias, name=name and name + "out")(attention_output)

    attention_output = keras.layers.Dropout(output_dropout, name=name and name + "output_dropout")(attention_output) if output_dropout > 0 else attention_output

    return attention_output

# Define U-Net model with custom attention
def unet_with_attention(input_shape=(384, 384, 3), num_classes=1):
    inputs = Input(shape=input_shape)
    c1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(64, 3, activation='relu', padding='same')(c1)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling2D(pool_size=(2, 2))(c1)

    c2 = Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(128, 3, activation='relu', padding='same')(c2)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling2D(pool_size=(2, 2))(c2)

    c3 = Conv2D(256, 3, activation='relu', padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(256, 3, activation='relu', padding='same')(c3)
    c3 = BatchNormalization()(c3)
    p3 = MaxPooling2D(pool_size=(2, 2))(c3)

    c4 = Conv2D(512, 3, activation='relu', padding='same')(p3)
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(512, 3, activation='relu', padding='same')(c4)
    c4 = BatchNormalization()(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(1024, 3, activation='relu', padding='same')(p4)
    c5 = BatchNormalization()(c5)
    c5 = Conv2D(1024, 3, activation='relu', padding='same')(c5)
    c5 = BatchNormalization()(c5)

    u6 = Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(512, 3, activation='relu', padding='same')(u6)
    c6 = BatchNormalization()(c6)
    c6 = Conv2D(512, 3, activation='relu', padding='same')(c6)
    c6 = BatchNormalization()(c6)

    u7 = Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(256, 3, activation='relu', padding='same')(u7)
    c7 = BatchNormalization()(c7)
    c7 = Conv2D(256, 3, activation='relu', padding='same')(c7)
    c7 = BatchNormalization()(c7)

    u8 = Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(128, 3, activation='relu', padding='same')(u8)
    c8 = BatchNormalization()(c8)
    c8 = Conv2D(128, 3, activation='relu', padding='same')(c8)
    c8 = BatchNormalization()(c8)

    u9 = Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(64, 3, activation='relu', padding='same')(u9)
    c9 = BatchNormalization()(c9)
    c9 = Conv2D(64, 3, activation='relu', padding='same')(c9)
    c9 = BatchNormalization()(c9)

    outputs = Conv2D(num_classes, 1, activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# Compile and train the model
def train_unet_model(model, train_data, val_data, epochs, batch_size, checkpoint_file):
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    callbacks = [
        ModelCheckpoint(checkpoint_file, save_best_only=True),
        TensorBoard(log_dir='./logs')
    ]

    history = model.fit(
        train_data[0], train_data[1],
        validation_data=val_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )
    
    return history

# Main function to run the training
def main():
    model = unet_with_attention()
    train_data = (TrainX, TrainY)
    val_data = (TestX, TestY)
    
    history = train_unet_model(
        model,
        train_data,
        val_data,
        epochs=config["EPOCHS"],
        batch_size=config["BATCH_SIZE"],
        checkpoint_file=config["CHECKPOINT_FILE"]
    )

    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('accuracy_plot.png')

    # Save the model summary to a file
    with open(config["MODEL_SUMMARY_FILE"], 'w') as f:
        with redirect_stdout(f):
            model.summary()

if __name__ == '__main__':
    main()
