import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import threading
import zlib

import numpy as np
import ConfigSpace.hyperparameters as CSH
from ConfigSpace import Configuration
from evosynth.config import get_settings
from evosynth.seeding import derive_int
from evosynth.models.cnn import DynamicCNN
from evosynth.search_spaces.nas import get_image_nas_search_space


class ImageCNNEvaluator:
    """
    Neural Architecture Search (NAS) and hyperparameter evaluator for image classification.
    Trains and validates dynamic PyTorch CNNs on a registered image dataset
    ([experiments.nas].dataset, default Fashion-MNIST). The search trains on search_train_rows
    and validates on search_val_rows, both stratified samples of the dataset's training images;
    the official test images are held out.
    """
    def __init__(self, space=None, epochs=None, batch_size=None, device=None, dataset=None):
        # Defaults come from [experiments.nas] (settings/experiments.toml) and
        # [datasets.<name>] (settings/datasets.toml)
        from evosynth import datasets

        s = get_settings()
        nas = s.experiments.nas
        self.dataset = nas.dataset if dataset is None else dataset
        spec = datasets.get_spec(self.dataset, s, kind="image")
        self.space = space or get_image_nas_search_space()
        self.epochs = nas.epochs if epochs is None else epochs
        self.batch_size = nas.batch_size if batch_size is None else batch_size
        self.sgd_momentum = nas.sgd_momentum
        device = nas.device if device is None else device
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")

        print(f"  [DATASET] Loading {spec.display_name}...")
        X, y = datasets.load(self.dataset, s)["train"]
        X_s, y_s = datasets.sample(X, y, spec.search_train_rows + spec.search_val_rows, s.run.data_seed)
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X_s, y_s, test_size=spec.search_val_rows, random_state=s.run.data_seed, stratify=y_s)
        self.in_channels = int(X_train.shape[1])
        self.num_classes = int(len(np.unique(y)))
        print(f"  [DATASET] Search: {len(y_train):,} training / {len(y_val):,} validation images "
              f"(test images held out), {self.num_classes} classes")

        # Training batches are shuffled per evaluation with a seeded generator (see evaluate)
        self.train_set = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
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

        # Phase 0 seeding: torch's generator is global, and evaluations run in parallel threads,
        # so training is serialised and seeded from run.seed + this configuration's vector.
        # The same configuration therefore always gets the same weights, dropout masks and
        # batch order, whatever order the evaluations run in.
        eval_seed = derive_int("nas", "evaluate", zlib.crc32(np.asarray(individual.vector, dtype=np.float64).tobytes()))
        with ImageCNNEvaluator._torch_lock:
            return self._train_and_score(eval_seed, conv1_ch, conv2_ch, fc_units, dropout, lr, opt_name, activation)

    _torch_lock = threading.Lock()

    def _train_and_score(self, eval_seed, conv1_ch, conv2_ch, fc_units, dropout, lr, opt_name, activation):
        torch.manual_seed(eval_seed)
        g = torch.Generator()
        g.manual_seed(eval_seed)
        train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, generator=g)

        model = DynamicCNN(
            in_channels=self.in_channels,
            conv1_ch=conv1_ch,
            conv2_ch=conv2_ch,
            fc_units=fc_units,
            dropout=dropout,
            activation=activation,
            num_classes=self.num_classes
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
            for images, labels in train_loader:
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
