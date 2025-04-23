import sys
import torch
from torch import nn
from torch.utils.data import DataLoader

from config import EPOCHS, BATCH_SIZE, DEVICE, DELTA, PATIENCE
from course_dataset import CourseDataset


class CourseRecommender(nn.Module):
    def __init__(
        self,
        num_users,
        num_courses,
        embedding_size=128,
        hidden_dim=256,
        dropout=0.1,
    ):
        super(CourseRecommender, self).__init__()

        self.user_embedding = nn.Embedding(
            num_embeddings=num_users, embedding_dim=embedding_size
        )
        self.course_embedding = nn.Embedding(
            num_embeddings=num_courses, embedding_dim=embedding_size
        )

        input_dim = 2 * embedding_size + 1

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, users, courses, watch_percentage):
        user_embedded = self.user_embedding(users)
        course_embedded = self.course_embedding(courses)
        watch_percentage = watch_percentage.unsqueeze(1)

        x = torch.cat([user_embedded, course_embedded, watch_percentage], dim=1)
        output = self.model(x)

        return output


def log_progress(
    epoch: int,
    step: int,
    total_loss: float,
    log_progress_step: int,
    data_size: int,
    losses: list,
):
    avg_loss = total_loss / log_progress_step
    sys.stderr.write(
        f"\r{epoch + 1:02d}/{EPOCHS:02d} | Step: {step}/{data_size} | Avg Loss: {avg_loss:<6.9f}"
    )
    sys.stderr.flush()
    losses.append(avg_loss)


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_func: torch.nn.Module,
    train_dataset: CourseDataset,
    valid_dataset: CourseDataset,
):
    total_loss = 0

    log_progress_step = 100
    losses = []

    # Variables para early stopping
    best_val_loss = float("inf")
    early_stop_counter = 0
    best_model_state = None

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    val_loader = DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    model.train()
    for e in range(EPOCHS):
        step_count = 0  # Reiniciar contador de pasos por época
        epoch_loss = 0

        # Entrenamiento
        for i, train_data in enumerate(train_loader):
            users = train_data["users"].to(DEVICE)
            courses = train_data["courses"].to(DEVICE)
            watch_percentage = train_data["watch_percentage"].to(DEVICE)
            ratings = (
                train_data["ratings"].to(DEVICE).squeeze()
            )  # Asegurarse de que ratings tenga la forma correcta

            output = model(users, courses, watch_percentage)
            output = output.squeeze()  # Eliminar dimensión extra
            loss = loss_func(output, ratings)

            total_loss += loss.item()
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step_count += len(train_data["users"])

            # Registro de progreso
            if (
                step_count % log_progress_step < BATCH_SIZE
                or i == len(train_loader) - 1
            ):
                log_progress(
                    e,
                    step_count,
                    total_loss,
                    log_progress_step,
                    len(train_dataset),
                    losses,
                )
                total_loss = 0

        # Evaluación en conjunto de validación después de cada época
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_data in val_loader:
                users = val_data["users"].to(DEVICE)
                courses = val_data["courses"].to(DEVICE)
                watch_percentage = val_data["watch_percentage"].to(DEVICE)
                ratings = val_data["ratings"].to(DEVICE).squeeze()

                output = model(users, courses, watch_percentage)
                output = output.squeeze()
                loss = loss_func(output, ratings)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"\nValidation Loss: {avg_val_loss:<6.9f}")

        # Early stopping
        if avg_val_loss < best_val_loss - DELTA:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            # Guardar el mejor modelo
            best_model_state = model.state_dict().copy()
            print("Validation improved! Saving model...")
        else:
            early_stop_counter += 1
            print(
                f"Validation did not improve. Early stopping counter: {early_stop_counter}/{PATIENCE}"
            )

        if early_stop_counter >= PATIENCE:
            print(f"\nEarly stopping triggered after {e + 1} epochs")
            break

        # Volver a modo entrenamiento para la siguiente época
        model.train()

        # Cargar el mejor modelo
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print("Loaded best model based on validation loss")
