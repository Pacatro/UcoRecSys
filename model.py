import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import root_mean_squared_error


class CourseRec(nn.Module):
    """
    Recommendation model for courses.
    The model takes as input the number of users courses and returns the rating.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_size: int = 256,
        hidden_dim: int = 256,
        dropout_rate: float = 0.2,
    ):
        super(CourseRec, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.item_embedding = nn.Embedding(num_items, embedding_size)
        self.fc1 = nn.Linear(2 * embedding_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, user_ids, item_ids):
        user_vec = self.user_embedding(user_ids)
        item_vec = self.item_embedding(item_ids)
        combined_vec = torch.cat((user_vec, item_vec), dim=1)
        x = self.relu(self.fc1(combined_vec))
        x = self.dropout(x)
        rating = self.fc2(x)
        return rating


def train_model(
    model: nn.Module,
    users: torch.Tensor,
    courses: torch.Tensor,
    ratings: torch.Tensor,
    epochs: int = 100,
    batch_size: int = 32,
):
    """
    Trains the recommendation model.

    :param model: The model to train.
    :param users: The users tensor
    :param courses: The courses tensor
    :param ratings: The ratings tensor
    :param epochs: The number of epochs to train the model.
    :param batch_size: The batch size to use for training.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    dataset = TensorDataset(users, courses, ratings)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    print("Starting training...")

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_user, batch_course, batch_rating in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_user, batch_course)
            outputs = outputs.squeeze()
            loss = torch.sqrt(criterion(outputs, batch_rating))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)

        # if epoch % 10 == 0:
        #     print(f"Epoch {epoch + 1}/{epochs}, loss: {avg_loss:.4f}")
        print(f"Epoch {epoch + 1}/{epochs}, loss: {avg_loss:.4f}")


def evaluate_model(
    model: nn.Module, users: torch.Tensor, courses: torch.Tensor, ratings: torch.Tensor
) -> float:
    """
    Evaluates the recommendation model.

    :param model: The model to evaluate.
    :param users: The users test tensor
    :param courses: The courses test tensor
    :param ratings: The ratings test tensor
    :return: The loss (RMSE) of the model.
    """
    model.eval()
    with torch.no_grad():
        y_pred = model(users, courses)
        loss = root_mean_squared_error(y_pred, ratings)
    return loss
