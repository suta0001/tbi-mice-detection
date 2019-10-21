import sklearn.ensemble
import sklearn.neighbors
import tensorflow as tf


def get_baseline_convolutional_encoder(filters, embedding_dimension,
                                       input_shape=None, dropout=0.05):
    encoder = tf.keras.models.Sequential()
    layers = tf.keras.layers

    # Initial conv
    if input_shape is None:
        # In this case we are using the encoder as part of a siamese network
        # and the input shape will be determined
        # automatically based on the input shape of the siamese network
        encoder.add(layers.Conv1D(filters, 32, padding='same',
                                  activation='relu'))
    else:
        # In this case we are using the encoder to build a classifier network
        # and the input shape must be defined
        encoder.add(layers.Conv1D(filters, 32, padding='same',
                                  activation='relu', input_shape=input_shape))
    encoder.add(layers.BatchNormalization())
    encoder.add(layers.SpatialDropout1D(dropout))
    encoder.add(layers.MaxPool1D(4, 4))

    # Further convs
    encoder.add(layers.Conv1D(2*filters, 3, padding='same', activation='relu'))
    encoder.add(layers.BatchNormalization())
    encoder.add(layers.SpatialDropout1D(dropout))
    encoder.add(layers.MaxPool1D())

    encoder.add(layers.Conv1D(3 * filters, 3, padding='same',
                              activation='relu'))
    encoder.add(layers.BatchNormalization())
    encoder.add(layers.SpatialDropout1D(dropout))
    encoder.add(layers.MaxPool1D())

    encoder.add(layers.Conv1D(4 * filters, 3, padding='same',
                              activation='relu'))
    encoder.add(layers.BatchNormalization())
    encoder.add(layers.SpatialDropout1D(dropout))
    encoder.add(layers.MaxPool1D())

    encoder.add(layers.GlobalMaxPool1D())

    encoder.add(layers.Dense(embedding_dimension))

    return encoder


def get_ffnn3hl():
    # hardcoded for siamese with 64 features
    num_features = 64
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8 * num_features, input_dim=num_features,
                              activation=tf.nn.leaky_relu),
        tf.keras.layers.Dense(4 * num_features, activation=tf.nn.leaky_relu),
        tf.keras.layers.Dense(2 * num_features, activation=tf.nn.leaky_relu),
        tf.keras.layers.Dense(4, activation=tf.nn.softmax)
    ])
    # compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def get_ml_model(ml_model):
    if ml_model == 'ffnn3hl':
        model = get_ffnn3hl()
    elif ml_model == 'knn':
        model = sklearn.neighbors.KNeighborsClassifier()
    elif ml_model == 'rf':
        model = sklearn.ensemble.RandomForestClassifier(n_estimators=100)
    return model


def build_siamese_net(encoder, input_shape,
                      distance_metric='uniform_euclidean'):
    assert distance_metric in ('uniform_euclidean', 'weighted_euclidean',
                               'uniform_l1', 'weighted_l1',
                               'dot_product', 'cosine_distance',
                               'uni_euc_cont_loss')
    layers = tf.keras.layers
    K = tf.keras.backend
    input_1 = layers.Input(input_shape)
    input_2 = layers.Input(input_shape)

    encoded_1 = encoder(input_1)
    encoded_2 = encoder(input_2)

    if distance_metric == 'weighted_l1':
        # This is the distance metric used in the original one-shot paper
        # https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf
        embedded_distance = layers.Subtract()([encoded_1, encoded_2])
        embedded_distance = layers.Lambda(lambda x: K.abs(x))(
            embedded_distance)
        output = layers.Dense(1, activation='sigmoid')(embedded_distance)
    elif distance_metric == 'uniform_euclidean':
        # Simpler, no bells-and-whistles euclidean distance
        # Still apply a sigmoid activation on the euclidean distance however
        embedded_distance = layers.Subtract(name='subtract_embeddings')(
            [encoded_1, encoded_2])
        # Sqrt of sum of squares
        embedded_distance = layers.Lambda(
            lambda x: K.sqrt(K.sum(K.square(x), axis=-1, keepdims=True)),
            name='euclidean_distance'
        )(embedded_distance)
        output = layers.Dense(1, activation='sigmoid')(embedded_distance)
    elif distance_metric == 'cosine_distance':
        raise NotImplementedError
        # cosine_proximity = layers.Dot(axes=-1, normalize=True)(
        #   [encoded_1, encoded_2])
        # ones = layers.Input(tensor=K.ones_like(cosine_proximity))
        # cosine_distance = layers.Subtract()([ones, cosine_proximity])
        # output = layers.Dense(1, activation='sigmoid')(cosine_distance)
    elif distance_metric == 'uni_euc_cont_loss':
        embedded_distance = layers.Subtract(name='subtract_embeddings')(
            [encoded_1, encoded_2])
        output = layers.Lambda(
            lambda x: K.sqrt(K.sum(K.square(x), axis=1, keepdims=True)),
        )(embedded_distance)
    else:
        raise NotImplementedError

    siamese = tf.keras.models.Model(inputs=[input_1, input_2], outputs=output)

    return siamese
