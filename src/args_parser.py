from argparse import ArgumentParser
import config


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="ucorecsys",
        description="Herramienta para entrenar, evaluar, hacer inferencia o entrenar un modelo de recomendación.",
    )

    # 1) Grupo mutuamente excluyente para el modo de ejecución: inference, train, eval, surprise
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "-i",
        "--inference",
        metavar="MODEL_PATH",
        help=(
            "Ejecutar inferencia sobre un modelo ya entrenado. "
            "Se debe especificar la ruta al archivo del modelo."
        ),
    )
    mode_group.add_argument(
        "-t",
        "--train",
        action="store_true",
        help="Entrenar el modelo con los parámetros especificados.",
    )
    mode_group.add_argument(
        "-e",
        "--eval",
        action="store_true",
        help="Evaluar el modelo propuesto (usando los parámetros de dataset y cvtype).",
    )
    mode_group.add_argument(
        "-s",
        "--surprise",
        action="store_true",
        help="Ejecutar evaluación de algoritmos de Surprise (biblioteca Surprise).",
    )

    # 2) Grupo para opciones de datos / validación
    data_group = parser.add_argument_group("Opciones de datos / validación")
    data_group.add_argument(
        "-ds",
        "--dataset",
        choices=config.DATASETS_CHOICES,
        default=config.DATASET,
        help=f"Nombre del dataset a utilizar. Por defecto: '{config.DATASET}'.",
    )
    data_group.add_argument(
        "-cv",
        "--cvtype",
        choices=config.CV_TYPES_CHOICES,
        default=config.CV_TYPE,
        help=f"Tipo de validación cruzada. Por defecto: '{config.CV_TYPE}'.",
    )

    # 3) Grupo para opciones de entrenamiento
    train_group = parser.add_argument_group("Opciones de entrenamiento")
    train_group.add_argument(
        "--epochs",
        type=int,
        default=config.EPOCHS,
        help=f"Número de epochs para entrenamiento (por defecto: {config.EPOCHS}).",
    )
    # train_group.add_argument(
    #     "--lr",
    #     type=float,
    #     default=config.LR,
    #     help=f"Tasa de aprendizaje para entrenamiento (por defecto: {config.LR}).",
    # )
    train_group.add_argument(
        "--batch-size",
        type=int,
        default=config.BATCH_SIZE,
        help=f"Tamaño de batch para entrenamiento (por defecto: {config.BATCH_SIZE}).",
    )
    train_group.add_argument(
        "--output-model",
        metavar="MODEL_OUT",
        default=config.OUTPUT_MODEL_PATH,
        help=f"Ruta donde guardar el modelo entrenado (por defecto: {config.OUTPUT_MODEL_PATH}).",
    )
    train_group.add_argument(
        "--k_ranking",
        type=int,
        default=config.K,
        help=f"Parámetro k para el cálculo de métricas de ranking (por defecto: {config.K}).",
    )

    # 4) Grupo para opciones de evaluación
    eval_group = parser.add_argument_group("Opciones de evaluación")
    eval_group.add_argument(
        "--balance",
        action="store_true",
        default=config.BALANCE,
        help="Balancear los datos de entrenamiento y validación.",
    )
    eval_group.add_argument(
        "--k",
        type=int,
        default=config.K,
        help=f"Parámetro k para el cálculo de métricas de ranking (por defecto: {config.K}).",
    )
    eval_group.add_argument(
        "-ns",
        "--splits",
        type=int,
        default=config.K_FOLD,
        help=f"Número de divisiones para la validación cruzada (por defecto: {config.K_FOLD}).",
    )

    # 5) Grupo para opciones generales / misc
    misc_group = parser.add_argument_group("Opciones generales")
    misc_group.add_argument(
        "-p",
        "--plot",
        action="store_true",
        help="Generar gráficos de métricas durante la ejecución.",
        default=config.PLOT,
    )
    misc_group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Mostrar información detallada durante la ejecución.",
    )

    return parser
