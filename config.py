"""Configuracion central del proyecto REBIN."""
import os
from pathlib import Path

# Debe quedar definido ANTES de importar tensorflow en cualquier archivo
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

BASE_DIR = Path(__file__).resolve().parent  # carpeta donde estan los .py (no tocar)

# Dataset: una subcarpeta por clase. Por defecto la carpeta "dataset" junto a estos .py.
# Para usar otra ruta sin editar nada:  python entrenar.py --data "C:/ruta/fotos"
DATA_DIR = str(BASE_DIR / "dataset")

# Fotos propias (mismas subcarpetas por clase). Si la carpeta no existe, se ignora.
EXTRA_DIR = str(BASE_DIR / "fotos_propias")

OUT_DIR_A = str(BASE_DIR / "salida_A")  # aqui queda model.tflite + labels.txt

IMG_SIZE = (224, 224)
BATCH = 32
EPOCHS_HEAD = 10
EPOCHS_FINE = 10
UNFREEZE_LAST = 30
SEED = 42
VAL_SPLIT = 0.2
