"""Carga de datos (dataset + fotos propias), pesos por clase y aumento de datos."""
import pathlib

import numpy as np
import tensorflow as tf

from config import IMG_SIZE, BATCH, SEED, VAL_SPLIT, EXTRA_DIR

EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp"}


def _listar(carpeta):
    """Lista (ruta, clase) de una carpeta con una subcarpeta por clase."""
    items = []
    subs = sorted(p for p in pathlib.Path(carpeta).iterdir()
                  if p.is_dir() and not p.name.startswith((".", "_")))
    for sub in subs:
        for f in sorted(sub.rglob("*")):
            if f.is_file() and f.suffix.lower() in EXTENSIONES:
                items.append((str(f), sub.name))
    return items


def _leer(ruta, etiqueta):
    img = tf.io.decode_image(tf.io.read_file(ruta), channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)  # float32, 0..255, RGB
    return img, etiqueta


def _crear_dataset(rutas, etiquetas, entrenar):
    ds = tf.data.Dataset.from_tensor_slices((rutas, etiquetas))
    if entrenar:
        ds = ds.shuffle(len(rutas), seed=SEED, reshuffle_each_iteration=True)
    return ds.map(_leer, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH)


def cargar_datos(data_dir, extra_dir=EXTRA_DIR):
    """Junta el dataset original + tus fotos propias (si la carpeta existe) en un solo monton."""
    originales = _listar(data_dir)
    clases_orig = sorted({c for _, c in originales})
    if len(clases_orig) < 2:
        raise SystemExit(
            f"No encuentro fotos en: {data_dir}\n"
            "Esa carpeta debe tener una subcarpeta por clase (plastic, glass, metal...) con fotos adentro.\n"
            "Descarga el dataset (ver README) y dejalo en la carpeta 'dataset', o usa --data \"C:/ruta\".")

    hay_extra = bool(extra_dir) and pathlib.Path(extra_dir).is_dir()
    propias = _listar(extra_dir) if hay_extra else []
    print(f"Fotos del dataset: {len(originales)} | fotos propias: {len(propias)}")

    items = originales + propias
    class_names = sorted({clase for _, clase in items})
    indice = {c: i for i, c in enumerate(class_names)}
    rutas = np.array([r for r, _ in items])
    etiquetas = np.array([indice[c] for _, c in items], dtype=np.int32)

    orden = np.random.RandomState(SEED).permutation(len(rutas))
    n_val = int(len(rutas) * VAL_SPLIT)
    val_i, train_i = orden[:n_val], orden[n_val:]
    print(f"Entrenamiento: {len(train_i)} fotos | validacion: {len(val_i)} fotos")

    train_ds = _crear_dataset(rutas[train_i], etiquetas[train_i], entrenar=True)
    val_ds = _crear_dataset(rutas[val_i], etiquetas[val_i], entrenar=False)
    return train_ds, val_ds, class_names


def calcular_class_weight(train_ds, class_names):
    n_classes = len(class_names)
    y_train = np.concatenate([y.numpy() for _, y in train_ds])
    counts = np.bincount(y_train, minlength=n_classes)
    print("Fotos de entrenamiento por clase:", dict(zip(class_names, counts.tolist())))
    return {i: float(len(y_train) / (n_classes * max(c, 1))) for i, c in enumerate(counts)}


def crear_aumento():
    """Solo se usa al entrenar, NO queda dentro del .tflite."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomBrightness(0.1, value_range=(0, 255)),
        tf.keras.layers.RandomContrast(0.1),
    ])


def preparar_datasets(train_ds, val_ds):
    autotune = tf.data.AUTOTUNE
    augment = crear_aumento()
    train_aug = train_ds.map(lambda x, y: (augment(x, training=True), y),
                             num_parallel_calls=autotune).prefetch(autotune)
    val_pf = val_ds.prefetch(autotune)
    return train_aug, val_pf
