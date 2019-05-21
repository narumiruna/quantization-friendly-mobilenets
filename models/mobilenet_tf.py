from itertools import repeat
from typing import Iterable

import tensorflow as tf
from tensorflow import layers, nn


def _pair(x):
    if isinstance(x, Iterable):
        return x
    return tuple(repeat(x, 2))


def conv(x, filters, stride=1):
    x = layers.conv2d(x, filters, kernel_size=(3, 3), strides=_pair(stride), padding="same", use_bias=True)
    x = nn.relu(x)
    return x


def quantization_friendly_depthwise_separable_conv(x, filters, stride=1):
    x = layers.separable_conv2d(x, x.shape[-1], kernel_size=(3, 3), strides=_pair(stride), padding="same")
    x = layers.conv2d(x, filters, kernel_size=(1, 1), strides=(1, 1), padding="same", use_bias=False)
    x = layers.batch_normalization(x,)
    x = nn.relu(x)
    return x


def nearby_int(x):
    return int(round(x))


def mobilenet(x, num_classes=1000, width_mult=1.0, shallow=False):
    settings = [
        (32, 2),
        (64, 1),
        (128, 2),
        (128, 1),
        (256, 2),
        (256, 1),
        (512, 2),
    ]
    if not shallow:
        settings += [(512, 1)] * 5
    settings += [
        (1024, 2),
        (1024, 1),
    ]

    for i, (filters, stride) in enumerate(settings):
        if i == 0:
            x = conv(x, nearby_int(filters * width_mult), stride=stride)
        else:
            x = quantization_friendly_depthwise_separable_conv(x, nearby_int(filters * width_mult), stride)

    # avg pool
    x = layers.average_pooling2d(x, 7, strides=1)

    # linear
    x = layers.dense(x, num_classes)

    # softmax
    x = nn.softmax(x)

    return x


def main():
    x = tf.zeros(shape=[1, 224, 224, 3])
    y = mobilenet(x)
    print(y)


if __name__ == '__main__':
    main()
