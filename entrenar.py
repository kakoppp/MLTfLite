"""
REBIN - Entrenamiento (MobileNetV3-Large -> model.tflite)

Uso:
    python entrenar.py
    python entrenar.py --data "C:/ruta/dataset"
    python entrenar.py --epochs-head 1 --epochs-fine 1     # prueba rapida
    python entrenar.py --sin-preentrenado                  # prueba sin internet

Salida: carpeta salida_A (model.tflite, labels.txt)
"""
import config  # primero: fija TF_USE_LEGACY_KERAS antes de importar tensorflow

import argparse
import pathlib

import tensorflow as tf

from config import DATA_DIR, OUT_DIR_A, EPOCHS_HEAD, EPOCHS_FINE, SEED
from datos import cargar_datos, calcular_class_weight, preparar_datasets
from modelo import construir_modelo, compilar, congelar_base, descongelar_ultimas
from evaluar import evaluar
from exportar import exportar_tflite, verificar_tflite


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA_DIR)
    parser.add_argument("--epochs-head", type=int, default=EPOCHS_HEAD)
    parser.add_argument("--epochs-fine", type=int, default=EPOCHS_FINE)
    parser.add_argument("--sin-preentrenado", action="store_true",
                        help="NO usar pesos de ImageNet. Solo para probar el flujo sin internet.")
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data)
    if not data_dir.is_dir():
        raise SystemExit(f"No existe la carpeta de datos: {data_dir}\n"
                         "Descarga el dataset (ver README) y dejalo en la carpeta 'dataset', "
                         "o usa --data \"C:/ruta\".")

    tf.keras.utils.set_random_seed(SEED)

    # Datos
    train_ds, val_ds, class_names = cargar_datos(data_dir)
    print("Clases:", class_names)
    class_weight = calcular_class_weight(train_ds, class_names)
    train_aug, val_pf = preparar_datasets(train_ds, val_ds)

    # Modelo
    model, base = construir_modelo(len(class_names), preentrenado=not args.sin_preentrenado)
    early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

    # Etapa 1: solo la cabeza
    congelar_base(base)
    compilar(model, 1e-3)
    model.fit(train_aug, validation_data=val_pf, epochs=args.epochs_head,
              class_weight=class_weight, callbacks=[early])

    # Etapa 2: fine-tuning de las ultimas capas con lr baja
    descongelar_ultimas(base)
    compilar(model, 1e-5)
    model.fit(train_aug, validation_data=val_pf, epochs=args.epochs_fine,
              class_weight=class_weight, callbacks=[early])

    # Evaluacion y exportacion
    eval_images = evaluar(model, val_ds, class_names)
    tflite_bytes, out = exportar_tflite(model, class_names, OUT_DIR_A)
    verificar_tflite(tflite_bytes, model, eval_images, out)


if __name__ == "__main__":
    main()
