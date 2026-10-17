"""
REBIN - Clasificar una foto con el modelo entrenado (salida_A/model.tflite)

Uso:
    python predecir.py                      # te pide la ruta de la foto
    python predecir.py "C:/fotos/botella.jpg"
    python predecir.py foto.jpg --modelo salida_A
"""
import config  # primero: fija TF_USE_LEGACY_KERAS antes de importar tensorflow

import argparse
import pathlib

import numpy as np
import tensorflow as tf
from PIL import Image

from config import IMG_SIZE, OUT_DIR_A


def cargar_modelo(carpeta):
    carpeta = pathlib.Path(carpeta)
    modelo = carpeta / "model.tflite"
    labels = carpeta / "labels.txt"
    if not modelo.is_file() or not labels.is_file():
        raise SystemExit(f"No encuentro model.tflite / labels.txt en {carpeta}\nPrimero corre: python entrenar.py")
    interp = tf.lite.Interpreter(model_path=str(modelo))
    interp.allocate_tensors()
    clases = labels.read_text(encoding="utf-8").splitlines()
    return interp, clases


def preparar_foto(ruta):
    img = Image.open(ruta).convert("RGB").resize(IMG_SIZE, Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)[None]  # 0..255, el modelo ya normaliza


def clasificar(interp, clases, ruta, top=3):
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    interp.set_tensor(inp["index"], preparar_foto(ruta))
    interp.invoke()
    probs = interp.get_tensor(out["index"])[0]
    orden = np.argsort(probs)[::-1][:top]
    return [(clases[i], float(probs[i])) for i in orden]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("foto", nargs="?", help="ruta de la foto")
    parser.add_argument("--modelo", default=OUT_DIR_A, help="carpeta con model.tflite y labels.txt")
    args = parser.parse_args()

    ruta = args.foto or input("Ruta de la foto: ")
    ruta = ruta.strip().strip('"').strip("'")  # por si la arrastras a la consola
    if not pathlib.Path(ruta).is_file():
        raise SystemExit(f"No existe la foto: {ruta}")

    interp, clases = cargar_modelo(args.modelo)
    resultados = clasificar(interp, clases, ruta)

    print(f"\nResultado: {resultados[0][0]}  ({resultados[0][1]:.1%})")
    print("\nTop 3:")
    for nombre, p in resultados:
        print(f"  {nombre:<25} {p:.1%}")


if __name__ == "__main__":
    main()
