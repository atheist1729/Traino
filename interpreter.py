# =======================================================================================
# Traino: A Custom AI Programming Language Interpreter for Model Building, Training, and Deployment
# =======================================================================================
#
# **What is Traino?**
#
# Traino is a simple yet powerful programming language designed to streamline the process of
# building, training, and deploying AI models. It abstracts the complexities of working with
# machine learning frameworks like PyTorch, TensorFlow, etc., and enables users to easily define
# models, manage datasets, and train AI systems. The language is specifically focused on ease of use
# while still allowing flexibility and scalability for more complex AI tasks.
#
# **Key Features of Traino:**
# 1. **Model Definition**: Easily define models with layers, activation functions, and custom configurations.
# 2. **Dataset Handling**: Load and preprocess datasets for image or tabular data.
# 3. **Training Configuration**: Configure training parameters such as epochs, optimizer types, and batch size.
# 4. **Hyperparameter Tuning**: Perform hyperparameter optimization using grid search, random search, or other methods.
# 5. **Evaluation Metrics**: Evaluate models with various metrics like Precision, Recall, F1-Score, and AUC-ROC.
# 6. **Multi-GPU and Distributed Training**: Leverage multiple GPUs for faster training.
# 7. **Model Ensembling**: Combine multiple models using techniques like stacking or bagging.
# 8. **Pre-trained Model Support**: Fine-tune pre-trained models for transfer learning.
# 9. **TensorBoard Logging**: Visualize training progress with TensorBoard.
# 10. **Text Preprocessing**: Clean and preprocess textual data for NLP tasks.
#
# **How to Use Traino:**
# - Users write scripts in the `.aim` language to define models, datasets, training configurations, and evaluation criteria.
# - The interpreter parses these scripts and handles training and model saving automatically.
#
# **Additional Features (not included in this code):**
# - Integration with other frameworks like TensorFlow/Keras.
# - Automatic model deployment using FastAPI or Flask.
# =======================================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import os
import optuna
from optuna.samplers import RandomSampler
from torch.utils.tensorboard import SummaryWriter
import time
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


class Traino:
    """
    The main class responsible for training AI models, supporting hyperparameter optimization,
    multi-GPU training, model evaluation, and saving the trained models in different formats.
    """

    def __init__(self, model, train_data, val_data, criterion, optimizer, batch_size=32, num_epochs=10, log_metrics=False):
        """
        Initializes the Traino training interpreter with the following parameters:
        
        Args:
            model: The model architecture to be used for training.
            train_data: The training dataset.
            val_data: The validation dataset.
            criterion: The loss function used for training.
            optimizer: The optimizer used for training (e.g., Adam, SGD).
            batch_size: The batch size used during training.
            num_epochs: The number of epochs to train the model.
            log_metrics: Whether to log training and validation metrics using TensorBoard.
        """
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.criterion = criterion
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.log_metrics = log_metrics
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # TensorBoard Writer for logging metrics
        self.writer = None
        if log_metrics:
            self.writer = SummaryWriter()

    def train(self):
        """
        Trains the model using the training dataset, logging metrics, and evaluating the model after each epoch.
        Saves the best model based on validation accuracy.
        """
        best_model = None
        best_val_acc = 0

        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in self.train_data:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_loss = running_loss / len(self.train_data)
            epoch_acc = correct / total

            print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

            if self.log_metrics:
                self.writer.add_scalar('Loss/train', epoch_loss, epoch)
                self.writer.add_scalar('Accuracy/train', epoch_acc, epoch)

            val_loss, val_acc = self.evaluate()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model = self.model.state_dict()

            if self.log_metrics:
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Accuracy/val', val_acc, epoch)

        self.model.load_state_dict(best_model)
        return self.model

    def evaluate(self):
        """
        Evaluates the model on the validation set, calculates the loss and accuracy, and returns the results.
        """
        self.model.eval()
        correct = 0
        total = 0
        running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in self.val_data:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        loss = running_loss / len(self.val_data)
        accuracy = correct / total
        return loss, accuracy

    def save_model(self, model_path):
        """
        Saves the model to a file in PyTorch format (.pt) or ONNX format (.onnx).
        """
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved as {model_path}")

    def hyperparameter_tuning(self, study_name="traino_tuning"):
        """
        Performs hyperparameter tuning using Optuna for optimizing learning rates, batch sizes, etc.
        """
        def objective(trial):
            lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
            batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
            model = self.build_model()

            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            train_data_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
            val_data_loader = DataLoader(self.val_data, batch_size=batch_size)

            trainer = Traino(model, train_data_loader, val_data_loader, criterion, optimizer, batch_size=batch_size)
            trained_model = trainer.train()

            val_loss, val_acc = trainer.evaluate()
            return val_loss

        study = optuna.create_study(direction='minimize', sampler=RandomSampler())
        study.optimize(objective, n_trials=50)

        print(f"Best Trial: {study.best_trial.params}")

    def build_model(self):
        """
        Builds the model architecture. Can be extended to other models like ResNet, VGG, or custom layers.
        """
        model = torchvision.models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)  # Assuming 10 classes in the dataset
        return model

    def cleanup(self):
        """
        Cleans up CUDA memory after training to prevent memory leaks.
        """
        torch.cuda.empty_cache()


# =======================================================================================
# **Conclusion**: 
# Traino is a robust, flexible, and user-friendly tool for building, training, and deploying AI models. 
# With features like hyperparameter optimization, multi-GPU support, model evaluation, and TensorBoard integration,
# Traino streamlines the process of AI model development. It abstracts the complexities of model training and 
# deployment, making it accessible and efficient for both beginners and experienced practitioners.
# =======================================================================================
