def generate_DeepConvLSTM_model(
        x_shape, class_number, filters, lstm_dims, learning_rate=0.01,
        regularization_rate=0.01, metrics=['accuracy']):
    """
    Generate a model with convolution and LSTM layers.
    See Ordonez et al., 2016, http://dx.doi.org/10.3390/s16010115

    Parameters
    ----------
    x_shape : tuple
        Shape of the input dataset: (num_samples, num_timesteps, num_channels)
    class_number : int
        Number of classes for classification task
    filters : list of ints
        number of filters for each convolutional layer
    lstm_dims : list of ints
        number of hidden nodes for each LSTM layer
    learning_rate : float
        learning rate
    regularization_rate : float
        regularization rate
    metrics : list
        Metrics to calculate on the validation set.
        See https://keras.io/metrics/ for possible values.

    Returns
    -------
    model : Keras model
        The compiled Keras model
    """
    dim_length = x_shape[1]  # number of samples in a time series
    dim_channels = x_shape[2]  # number of channels
    output_dim = class_number  # number of classes
    weightinit = 'lecun_uniform'  # weight initialization
    model = Sequential()  # initialize model
    model.add(BatchNormalization(input_shape=(dim_length, dim_channels)))
    # reshape a 2 dimensional array per file/person/object into a
    # 3 dimensional array
    model.add(
        Reshape(target_shape=(dim_length, dim_channels, 1)))
    for filt in filters:
        # filt: number of filters used in a layer
        # filters: vector of filt values
        model.add(
            Convolution2D(filt, kernel_size=(3, 1), padding='same',
                          kernel_regularizer=l2(regularization_rate),
                          kernel_initializer=weightinit))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
    # reshape 3 dimensional array back into a 2 dimensional array,
    # but now with more dept as we have the the filters for each channel
    model.add(Reshape(target_shape=(dim_length, filters[-1] * dim_channels)))

    for lstm_dim in lstm_dims:
        model.add(LSTM(units=lstm_dim, return_sequences=True,
                       activation='tanh'))

    model.add(Dropout(0.5))  # dropout before the dense layer
    # set up final dense layer such that every timestamp is given one
    # classification
    model.add(
        TimeDistributed(
            Dense(units=output_dim, kernel_regularizer=l2(regularization_rate))))
    model.add(Activation("softmax"))
    # Final classification layer - per timestep
    model.add(Lambda(lambda x: x[:, -1, :], output_shape=[output_dim]))

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=learning_rate),
                  metrics=metrics)

    return model