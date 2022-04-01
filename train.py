import os
import cv2
import numpy as np
from glob import glob
from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision

from models.unet import get_unet_model
from metrics import dice_loss, dice_coef, iou

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

IMG_HEIGHT = 512
IMG_WIDTH = 512
AUTO = tf.data.AUTOTUNE


def create_dir(path):
    """Create a directory."""
    if not os.path.exists(path):
        os.makedirs(path)


def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y


def load_data(path):
    x = sorted(glob(os.path.join(path, "image", "*.jpg")))
    y = sorted(glob(os.path.join(path, "mask", "*.jpg")))
    return x, y


def preprocess_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = x / 255.0
    x = x.astype(np.float32)
    return x


def preprocess_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = x / 255.0
    x = x > 0.5
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)
    return x


def preprocess_data(x, y):
    def _parse(x, y):
        x = preprocess_image(x)
        y = preprocess_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([IMG_HEIGHT, IMG_WIDTH, 3])
    y.set_shape([IMG_HEIGHT, IMG_WIDTH, 1])
    return x, y


def load_dataset(x, y, batch_size=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = (
        dataset.map(preprocess_data, num_parallel_calls=AUTO)
        .batch(batch_size)
        .prefetch(AUTO)
    )
    return dataset


if __name__ == "__main__":
    """Seeding"""
    SEEDS = 42
    np.random.seed(SEEDS)
    tf.random.set_seed(SEEDS)

    """ Directory for storing files """
    create_dir("files")

    """ Hyperparameters """
    batch_size = 2
    lr = 1e-4
    num_epochs = 5
    model_path = os.path.join("files", "model.h5")
    csv_path = os.path.join("files", "data.csv")

    """ Dataset """
    dataset_path = os.path.join("new_data")
    train_path = os.path.join(dataset_path, "train")
    valid_path = os.path.join(dataset_path, "valid")

    train_x, train_y = load_data(train_path)
    train_x, train_y = shuffling(train_x, train_y)
    valid_x, valid_y = load_data(valid_path)

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Valid: {len(valid_x)} - {len(valid_y)}")

    train_dataset = load_dataset(train_x, train_y, batch_size=batch_size)
    valid_dataset = load_dataset(valid_x, valid_y, batch_size=batch_size)

    """ Model """
    model = get_unet_model((IMG_HEIGHT, IMG_WIDTH, 3))
    metrics = [dice_coef, iou, Recall(), Precision()]
    model.compile(loss=dice_loss, optimizer=Adam(lr), metrics=metrics)

    """Setting up Training Callbacks"""
    train_callbacks = [
        tf.keras.callbacks.ModelCheckpoint(model_path, verbose=1, save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.1, patience=10, min_lr=1e-7, verbose=1
        ),
        tf.keras.callbacks.CSVLogger(csv_path),
        tf.keras.callbacks.TensorBoard(),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=False
        ),
    ]

    model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=valid_dataset,
        callbacks=train_callbacks,
        shuffle=False
    )
