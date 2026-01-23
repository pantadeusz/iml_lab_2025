import keras

def create_model_with_params(
    encoder,
    activation='relu',
    use_batch_norm=True,
    dense_shapes=[64],
    bidirectional_shapes=[64],
    dropout_rate=0.5,
    learning_rate=1e-3,
    optimizer='adam'
):
    """
    Creates a semi flexible model compatible with keras tuner.
    
    Args:
        hp: HyperParameters object from Keras Tuner
        input_shape: Input image shape
        num_classes: Number of output classes
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

    # Encoder layer
    model.add(encoder)

    # Embedding layer
    model.add(keras.layers.Embedding(input_dim=len(encoder.get_vocabulary()), output_dim=bidirectional_shapes[0], mask_zero=True))

    # Bidirectional layers
    for i, shape in enumerate(bidirectional_shapes):

        is_last_bi_layer = (i == len(bidirectional_shapes) - 1)
        return_sequence = not is_last_bi_layer
        model.add(keras.layers.Bidirectional(keras.layers.LSTM(shape, return_sequences=return_sequence)))

    # Dense layers
    for shape in dense_shapes:
        model.add(keras.layers.Dense(
            shape,
            activation=activation,
            kernel_initializer='glorot_uniform'
        ))
        if use_batch_norm:
            model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(dropout_rate))

    # Output layer
    model.add(keras.layers.Dense(1))

    # Compile model
    if optimizer == 'adam':
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        opt = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=opt, # Lsp can be wrong here because the compile method seems to mislead with its parameters. It can take the optimizer object
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    return model

