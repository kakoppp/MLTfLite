"""Evaluacion del modelo sobre validacion."""
import numpy as np


def evaluar(model, val_ds, class_names):
    n_classes = len(class_names)
    y_true, y_pred, eval_images = [], [], []
    for xb, yb in val_ds:
        pb = model.predict_on_batch(xb)
        y_true.append(yb.numpy())
        y_pred.append(np.argmax(pb, axis=1))
        if len(eval_images) < 16:
            eval_images.extend(xb.numpy()[:16])
    y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)

    try:
        from sklearn.metrics import classification_report, confusion_matrix
        print("\n=== Reporte por clase (validacion) ===")
        print(classification_report(y_true, y_pred, labels=range(n_classes),
                                    target_names=class_names, zero_division=0))
        print("Matriz de confusion (filas=real, columnas=predicho):")
        print(confusion_matrix(y_true, y_pred, labels=range(n_classes)))
    except ImportError:
        print("Instala scikit-learn para ver el reporte por clase. Accuracy:",
              float((y_true == y_pred).mean()))
    return eval_images