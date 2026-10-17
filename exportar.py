"""Exportacion a TFLite y verificacion contra Keras."""
import pathlib
import time

import numpy as np
import tensorflow as tf


def exportar_tflite(model, class_names, out_dir):
    out = pathlib.Path(out_dir)
    out.mkdir(exist_ok=True)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]  # cuantizacion float16
    tflite_bytes = converter.convert()
    (out / "model.tflite").write_bytes(tflite_bytes)
    # write_bytes: mismo salto de linea (\n) en Windows, Linux y Mac
    (out / "labels.txt").write_bytes("\n".join(class_names).encode("utf-8"))
    print(f"\nmodel.tflite: {len(tflite_bytes) / 1e6:.2f} MB (meta del proyecto: <= 10 MB)")
    return tflite_bytes, out


def verificar_tflite(tflite_bytes, model, eval_images, out):
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inp, outp = interp.get_input_details()[0], interp.get_output_details()[0]

    images = np.array(eval_images[:16])
    tfl_probs, times = [], []
    for img in images:
        interp.set_tensor(inp["index"], img[None].astype(np.float32))
        t0 = time.perf_counter()
        interp.invoke()
        times.append((time.perf_counter() - t0) * 1000)
        tfl_probs.append(interp.get_tensor(outp["index"])[0])
    tfl_probs = np.array(tfl_probs)
    keras_probs = model.predict(images, verbose=0)

    print(f"Coincidencia de clase Keras vs TFLite: {(tfl_probs.argmax(1) == keras_probs.argmax(1)).mean():.0%}")
    print(f"Diferencia maxima en probabilidades: {np.abs(tfl_probs - keras_probs).max():.4f}")
    print(f"Latencia en ESTE computador: {np.median(times):.1f} ms por foto "
          f"(la medida que importa es en un celular Android de gama media)")
    print(f"Archivos guardados en: {out.resolve()}")
