import tensorflow as tf


def load_mnist(train_size=3000, val_size=0, test_size=1000):
    """Load MNIST, normalize, and return train/val/test arrays with requested sizes.

    Args:
        train_size: Number of samples to use for training (taken from the MNIST training split).
        val_size: Number of samples to use for validation (taken from the remaining MNIST training split).
        test_size: Number of samples to use for testing (taken from the MNIST test split).

    Returns:
        (train_images, train_labels, val_images, val_labels, test_images, test_labels)
    """
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    # Normalize to [0,1]
    train_images = train_images.astype("float32") / 255.0
    test_images = test_images.astype("float32") / 255.0

    # Safety: clamp requested sizes to available data
    total_train = train_images.shape[0]
    t_size = max(0, min(train_size, total_train))
    v_size = max(0, min(val_size, max(0, total_train - t_size)))
    te_size = max(0, min(test_size, test_images.shape[0]))

    # Split training into train and validation
    t_imgs = train_images[:t_size]
    t_lbls = train_labels[:t_size]
    v_imgs = train_images[t_size:t_size + v_size]
    v_lbls = train_labels[t_size:t_size + v_size]

    # Limit test set
    te_imgs = test_images[:te_size]
    te_lbls = test_labels[:te_size]

    return t_imgs, t_lbls, v_imgs, v_lbls, te_imgs, te_lbls


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

