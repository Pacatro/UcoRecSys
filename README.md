# UcoRecSys

Este repositorio forma parte de Trabajo de Fin de Grado (TFG) realizado por Franacisco de Paula Algar Muñoz en la Universidad de Córdoba (UCO) titulado: ***Aplicación de sistemas de recomendación en entornos educativos.***

El objetivo de este TFG es desarrollar un sistema de recomendación para e-learning basado en un conjunto de datos de referencia, lo que permitirá evaluar su rendimiento en comparación con otros modelos previos.

## Cómo usar

```bash
usage: ucorecsys [-h] (-i MODEL_PATH | -t | -e | -s) [-ds {mars,coursera,itm}] [-cv {kfold,loo}] [--epochs EPOCHS] [--batch-size BATCH_SIZE]
                 [--output-model MODEL_OUT] [--k_ranking K_RANKING] [--balance] [--k K] [-ns SPLITS] [-p] [-v]
```

### Opciones

```bash
options:
  -h, --help            show this help message and exit
  -i MODEL_PATH, --inference MODEL_PATH
                        Run inference on an already-trained model. You must specify the path to the model file.
  -t, --train           Train the model using the specified parameters.
  -e, --eval            Evaluate the proposed model (using the dataset and cvtype parameters).
  -s, --surprise        Run evaluation of Surprise algorithms (using the Surprise library).

Data / Validation Options:
  -ds {mars,coursera,itm}, --dataset {mars,coursera,itm}
                        Name of the dataset to use. Default: 'mars'.
  -cv {kfold,loo}, --cvtype {kfold,loo}
                        Type of cross-validation. Default: 'kfold'.

Training Options:
  --epochs EPOCHS       Number of epochs for training (default: 50).
  --batch-size BATCH_SIZE
                        Batch size for training (default: 128).
  --output-model MODEL_OUT
                        Path to save the trained model (default: model.pt).
  --k_ranking K_RANKING
                        Parameter k for ranking metrics calculation (default: 10).

Evaluation Options:
  --balance             Balance training and validation data.
  --k K                 Parameter k for ranking metrics calculation (default: 10).
  -ns SPLITS, --splits SPLITS
                        Number of splits for cross-validation (default: 10).

General Options:
  -p, --plot            Generate metric plots during execution.
  -v, --verbose         Print detailed information during execution.
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
    uv build
    ```

## Autor  

**Francisco de Paula Algar Muñoz**  

## Tutores  

- **Amelia Zafra Gómez**  
- **Cristóbal Romero Morales**
