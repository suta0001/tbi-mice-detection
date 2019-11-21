import sklearn.ensemble
import sklearn.neighbors
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import layers
from tensorflow.python.keras import Model
from tensorflow.python.keras import Sequential
from xgboost import XGBClassifier


class EuclideanDistance(layers.Layer):
    def __init__(self, **kwargs):
        super(EuclideanDistance, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 2:
            raise ValueError('Must have 2 inputs.')
        super(EuclideanDistance, self).build(input_shape)

    def call(self, inputs):
        return K.sqrt(K.sum(K.square(inputs[0] - inputs[1]),
                            axis=1, keepdims=True))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1, 1)


def get_baseline_convolutional_encoder(filters, embedding_dimension,
                                       input_shape=None, dropout=0.05):
    encoder = Sequential()
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
    model = Sequential([
        layers.Dense(8 * num_features, input_dim=num_features,
                     activation='relu'),
        layers.Dense(4 * num_features, activation='relu'),
        layers.Dense(2 * num_features, activation='relu'),
        layers.Dense(4, activation='softmax')
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
    elif ml_model == 'xgb':
        model = XGBClassifier()
    return model


def build_siamese_net(encoder, input_shape,
                      distance_metric='uniform_euclidean'):
    assert distance_metric in ('uniform_euclidean', 'weighted_euclidean',
                               'uniform_l1', 'weighted_l1',
                               'dot_product', 'cosine_distance',
                               'uni_euc_cont_loss')
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
        output = EuclideanDistance(name='distance')([encoded_1, encoded_2])
    else:
        raise NotImplementedError

    siamese = Model(inputs=[input_1, input_2], outputs=output)

    return siamese
