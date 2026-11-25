import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import ConfigSpace.hyperparameters as CSH
from ConfigSpace import Configuration
from evosynth.config import get_settings
from evosynth.models.cnn import DynamicCNN
from evosynth.search_spaces.nas import get_image_nas_search_space


class ImageCNNEvaluator:
    """
    Neural Architecture Search (NAS) and Hyperparameter Evaluator for non-tabular vision data.
    Trains and validates dynamic PyTorch CNN models on image classification (Handwritten Digits/MNIST).
    """
    def __init__(self, space=None, epochs=None, batch_size=None, device=None):
        # Defaults come from [experiments.nas] in settings/experiments.toml
        s = get_settings()
        nas = s.experiments.nas
        self.space = space or get_image_nas_search_space()
        self.epochs = nas.epochs if epochs is None else epochs
        self.batch_size = nas.batch_size if batch_size is None else batch_size
        self.sgd_momentum = nas.sgd_momentum
        device = nas.device if device is None else device
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")

        # Load image dataset (8x8 digits scaled to image tensors, zero-latency setup)
        from sklearn.datasets import load_digits
        from sklearn.model_selection import train_test_split

        raw = load_digits()
        X = raw.images[:, np.newaxis, :, :].astype(np.float32) / 16.0  # (N, 1, 8, 8) normalized
        y = raw.target.astype(np.int64)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=nas.validation_fraction, random_state=s.run.random_state, stratify=y
        )

        self.train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
            batch_size=self.batch_size,
            shuffle=True
        )
        self.val_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
            batch_size=self.batch_size,
            shuffle=False
        )

    def evaluate(self, individual):
        """
        Receives an Individual (continuous [0, 1] vector), decodes it into
        CNN architectural topology and training hyperparameters, trains the model,
        and returns validation accuracy.
        """
        valid_vec = np.copy(individual.vector)
        for i, hp_name in enumerate(list(self.space.keys())):
            hp = self.space.get_hyperparameter(hp_name)
            if isinstance(hp, CSH.CategoricalHyperparameter):
                num_choices = len(hp.choices)
                idx = int(np.floor(valid_vec[i] * num_choices))
                idx = min(idx, num_choices - 1)
                valid_vec[i] = float(idx)

        config = Configuration(self.space, vector=valid_vec)

        conv1_ch = int(config["conv1_channels"])
        conv2_ch = int(config["conv2_channels"])
        fc_units = int(config["fc_units"])
        dropout = float(config["dropout_rate"])
        lr = float(config["learning_rate"])
        opt_name = config["optimizer"]
        activation = config["activation"]

        model = DynamicCNN(
            in_channels=1,
            conv1_ch=conv1_ch,
            conv2_ch=conv2_ch,
            fc_units=fc_units,
            dropout=dropout,
            activation=activation,
            num_classes=10
        ).to(self.device)

        criterion = nn.CrossEntropyLoss()
        if opt_name == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=lr)
        elif opt_name == "RMSprop":
            optimizer = optim.RMSprop(model.parameters(), lr=lr)
        else:
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=self.sgd_momentum)

        # Train model for fixed epochs
        model.train()
        for _ in range(self.epochs):
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # Evaluate on validation set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = (correct / total) * 100.0 if total > 0 else 0.0
        return float(accuracy)
