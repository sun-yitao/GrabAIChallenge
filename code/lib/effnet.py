from keras.models import Model
from keras import layers


def get_post(x_in):
    x = layers.LeakyReLU()(x_in)
    x = layers.BatchNormalization()(x)
    return x


def get_block(x_in, ch_in, ch_out, regularizer):
    x = layers.Conv2D(ch_in,
                      kernel_size=(1, 1),
                      padding='same',
                      use_bias=False,
                      kernel_regularizer=regularizer)(x_in)
    x = get_post(x)

    x = layers.DepthwiseConv2D(kernel_size=(1, 3), 
                               padding='same', 
                               use_bias=False,
                               depthwise_regularizer=regularizer)(x)
    x = get_post(x)
    x = layers.MaxPool2D(pool_size=(2, 1),
                         strides=(2, 1))(x)  # Separable pooling

    x = layers.DepthwiseConv2D(kernel_size=(3, 1),
                               padding='same',
                               use_bias=False,
                               depthwise_regularizer=regularizer)(x)
    x = get_post(x)

    x = layers.Conv2D(ch_out,
                      kernel_size=(2, 1),
                      strides=(1, 2),
                      padding='same',
                      use_bias=False,
                      kernel_regularizer=regularizer)(x)
    x = get_post(x)

    return x


def Effnet(input_shape, nb_classes, include_top=True, weights=None, regularizer=1e-4):
    x_in = layers.Input(shape=input_shape)

    x = get_block(x_in, 32, 64, regularizer)
    x = get_block(x, 64, 128, regularizer)
    x = get_block(x, 128, 256, regularizer)

    if include_top:
        x = layers.Flatten()(x)
        x = layers.Dense(nb_classes, activation='softmax')(x)

    model = Model(inputs=x_in, outputs=x)

    if weights is not None:
        model.load_weights(weights, by_name=True)

    return model
