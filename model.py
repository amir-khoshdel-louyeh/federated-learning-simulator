import tensorflow as tf


def load_mnist(subset=3000, test_size=1000):
    """Load MNIST, normalize, and return trimmed train/test arrays.

    Returns: (train_images, train_labels, test_images, test_labels)
    """
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.astype("float32") / 255.0
    test_images = test_images.astype("float32") / 255.0
    return train_images[:subset], train_labels[:subset], test_images[:test_size], test_labels[:test_size]


def make_model():
    """Return a compiled Keras model for MNIST (28x28 grayscale).

    This central module avoids circular imports and lets other algorithms
    reuse the same model builder.
    """
    inputs = tf.keras.Input(shape=(28, 28))
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

