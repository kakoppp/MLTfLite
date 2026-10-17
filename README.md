# REBIN - Clasificador de basura (MobileNetV3 → TFLite)

Clasifica una foto en 9 clases: `battery`, `cardboard`, `clothes`, `electronic`, `glass`, `metal`, `paper`, `plastic`, `shoes`.

Requisitos: Python 3.11 (probado) e internet la primera vez (para instalar y bajar los pesos de ImageNet).

## 1. Instalación

```
git clone <URL_DEL_REPO>
cd testWhitTfLite
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

En Linux/Mac, el paso de activar es `source .venv/bin/activate`.

## 2. Probar el modelo (sin entrenar)

El repo ya trae el modelo entrenado en `salida_A/`. Para clasificar una foto:

```
python predecir.py
python predecir.py "C:\ruta\foto.jpg"
```

La primera te pide la ruta de la foto. Muestra la clase y el top 3.

## 3. Entrenar (opcional)

Para entrenar hace falta el dataset, que no está en el repo por su tamaño.

1. Descarga el dataset: `<LINK_DEL_DATASET>`
2. Descomprímelo como una carpeta `dataset` junto a los `.py`, con una subcarpeta por clase:

```
testWhitTfLite\
├── dataset\
│   ├── plastic\
│   ├── glass\
│   ├── metal\
│   └── ...
├── config.py
└── entrenar.py
```

3. Entrena:

```
python entrenar.py
```

Si el dataset está en otra ruta, no hace falta editar nada:

```
python entrenar.py --data "C:\ruta\al\dataset"
```

Prueba rápida (1 época por etapa):

```
python entrenar.py --epochs-head 1 --epochs-fine 1
```

La salida queda en `salida_A\` (`model.tflite` y `labels.txt`) y **sobrescribe** el modelo anterior. Para conservarlo, copia la carpeta antes (ej. `salida_A_v1`) y pruébala con:

```
python predecir.py --modelo salida_A_v1
```

## 4. Fotos propias (opcional)

Para sumar tus propias fotos al entrenamiento, crea una carpeta `fotos_propias` junto a los `.py`, con las **mismas subcarpetas** que el dataset (los nombres deben ser idénticos):

```
fotos_propias\
├── plastic\
├── glass\
└── otros\      ← una carpeta nueva se suma sola como una clase más
```

Se mezclan con el dataset al entrenar. Si la carpeta no existe, se ignora.

Las fotos para medir el modelo deben ir en otra carpeta (ej. `fotos_prueba`) que nunca entre al entrenamiento.

## Rutas

Las rutas están en `config.py` (`DATA_DIR` y `EXTRA_DIR`). Por defecto apuntan a las carpetas `dataset` y `fotos_propias` junto a los `.py`.
