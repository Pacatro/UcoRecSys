import torch

EPOCHS = 15
PATIENCE = 3  # Número de épocas para esperar mejora antes de detener el entrenamiento
DELTA = 0.0001  # Mejora mínima requerida para considerar progreso

BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
