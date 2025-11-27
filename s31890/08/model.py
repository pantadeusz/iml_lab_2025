import keras


def create_model_with_params(
    hp,
    input_shape,
    num_classes=3,
    num_conv_blocks=2,
    conv_filters=[32, 64, 128],
    kernel_size=(3, 3),
    activation="relu",
    use_batch_norm=True,
    num_dense_layers=3,
    units_per_dense=128,
    dropout_rate=0.5,
    learning_rate=1e-3,
    optimizer="adam",
):
    """
    Creates a semi flexible model compatible with keras tuner.

    Args:
        hp: HyperParameters object from Keras Tuner
        input_shape: Input image shape
        num_classes: Number of output classes
        num_conv_blocks: Number of Conv2D + MaxPooling blocks
        conv_filters: List of filters for each block
        kernel_size: Size of convolutional kernel
        activation: Activation function
        use_batch_norm: Whether to use batch normalization
        num_dense_layers: Number of dense layers
        units_per_dense: Number of units per dense layer
        dropout_rate: Dropout rate
        learning_rate: Learning rate for optimizer
        optimizer: Optimizer name ('adam', 'sgd', 'rmsprop')

    Returns:
        A compiled Keras model
    """
    model = keras.Sequential()

    # Input layer
    model.add(keras.layers.Input(shape=input_shape))
    # model.add(keras.layers.Rescaling(1./255))

    # Convolutional blocks
    for i in range(num_conv_blocks):
        filters = hp.Choice(f"conv_filters_{i}", conv_filters)
        model.add(
            keras.layers.Conv2D(
                filters,
                kernel_size,
                activation=activation,
                kernel_initializer="glorot_uniform",
            )
        )
        if use_batch_norm:
            model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling2D((2, 2)))

    # Flatten
    model.add(keras.layers.Flatten())

    # Dense layers
    for i in range(num_dense_layers):
        model.add(
            keras.layers.Dense(
                units_per_dense,
                activation=activation,
                kernel_initializer="glorot_uniform",
            )
        )
        if use_batch_norm:
            model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(dropout_rate))

    # Output layer
    model.add(keras.layers.Dense(num_classes, activation="softmax"))

    # Compile model
    if optimizer == "adam":
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == "sgd":
        opt = keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer == "rmsprop":
        opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        opt = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=opt,  # Lsp can be wrong here because the compile method seems to mislead with its parameters. It can take the optimizer object
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
