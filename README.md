# UcoRecSys

Este repositorio forma parte de Trabajo de Fin de Grado (TFG) realizado por Franacisco de Paula Algar Muñoz en la Universidad de Córdoba (UCO) titulado: ***Aplicación de sistemas de recomendación en entornos educativos*** ([PDF](./Memoria_TFG.pdf)).

El objetivo de este TFG es desarrollar un sistema de recomendación para e-learning basado en un conjunto de datos de referencia, lo que permitirá evaluar su rendimiento en comparación con otros modelos previos.

## Cómo usar

```bash
usage: ucorecsys [-h] (-i MODEL_PATH | -t | -e | -s | -st) [-ds {mars,itm}] [-cv {kfold,loo}]
                 [--top_k TOP_K] [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--output_model MODEL_OUT]
                 [-lr LR] [-k K_SPLITS] [--seeds SEEDS] [-p] [-v]
```

### Opciones

```bash
Tool to train, evaluate, or perform inference with a recommendation model.

options:
  -h, --help            show this help message and exit
  -i MODEL_PATH, --inference MODEL_PATH
                        Run inference on a trained model (provide path to model file).
  -t, --train           Train the model.
  -e, --eval            Evaluate the model.
  -s, --surprise        Run Surprise evaluation.
  -st, --stats_test     Run stats test (default: False).

Common Options:
  -ds {mars,itm}, --dataset {mars,itm}
                        Dataset to use (default: mars).
  -cv {kfold,loo}, --cvtype {kfold,loo}
                        Cross-validation type (default: kfold).
  --top_k TOP_K         Top-k value for ranking metrics (default: 10).

Training Options:
  --epochs EPOCHS       Training epochs (default: 50).
  --batch_size BATCH_SIZE
                        Training batch size (default: 128).
  --output_model MODEL_OUT
                        Path to save trained model (default: model.pt).
  -lr LR                Learning rate (default: 0.001).

Evaluation Options:
  -k K_SPLITS, --k_splits K_SPLITS
                        Number of CV splits (default: 5).
  --seeds SEEDS         Random seeds (default: [0, 1, 42]).

Miscellaneous Options:
  -p, --plot            Generate plots.
  -v, --verbose         Enable verbose output.```
```

## Puesta en marcha

> [!NOTE]
> Para ejecutar este proyecto es necesario tener instalado el manajador de paquetes [`uv`](https://docs.astral.sh/uv/)

Sigue estos pasos para ejecutar el proyecto:

1. **Clona el repositorio**

    ```bash
    git clone https://github.com/Pacatro/UcoRecSys.git
    cd UcoRecSys
    ```

2. **Ejecuta el proyecto**

    El siguiente comando creará el entorno virtual e instalando todas las dependencias necesarias:

    ```bash
    uv run src/main.py -h
    ```

## Autor  

**Francisco de Paula Algar Muñoz**  

## Tutores  

**Amelia Zafra Gómez**  
**Cristóbal Romero Morales**
