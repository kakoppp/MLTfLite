"""Definicion del modelo MobileNetV3-Large (arquitectura y compilacion)."""
import config  # fija TF_USE_LEGACY_KERAS antes de importar tensorflow

import tensorflow as tf

from config import IMG_SIZE, UNFREEZE_LAST


def construir_modelo(n_classes, preentrenado=True):
    # MobileNetV3 ya incluye su preprocesamiento: recibe pixeles 0..255
    base = tf.keras.applications.MobileNetV3Large(
        input_shape=IMG_SIZE + (3,), include_top=False,
        weights="imagenet" if preentrenado else None,
        include_preprocessing=True)

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,), name="foto")  # float32, 0..255, RGB
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax", name="probabilidades")(x)
    return tf.keras.Model(inputs, outputs), base


def compilar(model, lr):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"])


def congelar_base(base):
    base.trainable = False


def descongelar_ultimas(base, n=UNFREEZE_LAST):
    """Descongela las ultimas n capas; BatchNorm siempre queda congelada."""
    base.trainable = True
    for layer in base.layers[:-n]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
