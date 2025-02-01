
# **Traino: A Custom AI Programming Language for Building, Training, and Deploying Models**

## **Overview**

**Traino** is a high-level programming language designed to simplify the process of building, training, and deploying AI models. Traino abstracts away the complexities of working with popular machine learning frameworks like **PyTorch** and **TensorFlow** and provides an easy-to-use syntax to define, train, and evaluate machine learning models with minimal code. With a focus on flexibility, scalability, and ease of use, Traino is perfect for both beginners and experienced AI practitioners.

### **Key Features:**
- **Model Definition**: Define models using predefined layers such as Conv2D, Dense, Dropout, etc.
- **Dataset Handling**: Load and preprocess datasets (image, tabular, text) with built-in augmentations.
- **Training Configuration**: Configure the training loop, optimizer, epochs, batch size, and more.
- **Hyperparameter Tuning**: Perform optimization on hyperparameters like learning rate, batch size using **Optuna**.
- **Evaluation Metrics**: Evaluate the model using precision, recall, F1-score, AUC-ROC, etc.
- **Multi-GPU and Distributed Training**: Train models efficiently on multiple GPUs.
- **Model Saving**: Save the trained models in **PyTorch** (`.pt`) or **ONNX** (`.onnx`) formats.
- **Pre-trained Models**: Fine-tune pre-trained models (e.g., ResNet, VGG) on custom datasets.
- **Logging with TensorBoard**: Visualize training metrics in real-time using **TensorBoard**.

---

## **Installation**

### **Step 1: Clone the Repository**
First, clone the Traino repository from GitHub:

```bash
git clone https://github.com/your-repository/traino.git
cd traino
```

### **Step 2: Install Dependencies**

Traino requires **Python 3.6+** and several dependencies. Install them using `pip`:

```bash
pip install -r requirements.txt
```

This will install the necessary libraries such as **PyTorch**, **Optuna**, **TensorBoard**, **torchvision**, and others.

---

## **Getting Started with Traino**

### **Creating a `.aim` File**

Traino scripts are written in `.aim` files. These files contain definitions of models, datasets, training configurations, and evaluation metrics.

#### **Example `.aim` file**:

```aim
model SimpleCNN {
    Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(3, 224, 224));
    MaxPooling2D(pool_size=(2, 2));
    Dense(128, activation="relu");
    Dropout(0.5);
    Dense(10, activation="softmax");
};

dataset "data/cifar10" split(train=70%, val=15%, test=15%) augment rotate flip normalize;

train SimpleCNN on "data/cifar10" for 50 epochs using Adam(learning_rate=0.001) batch_size=64;

evaluate SimpleCNN using "data/cifar10_test" metrics=accuracy precision recall f1_score auc_roc;

save SimpleCNN as "simple_cnn_model.pt";
```

### **Running the `.aim` File**

Once you have written your `.aim` file, you can run it using the **Traino Interpreter**:

```bash
python interpreter.py train_model.aim
```

This will trigger the model training, evaluation, and saving process as defined in the `.aim` script.

---

## **Traino Syntax and Structure**

### **1. Model Definition**

To define a model, you can use layers like **Conv2D**, **Dense**, **MaxPooling2D**, **Dropout**, etc., in the following format:

```aim
model ModelName {
    Conv2D(filters, kernel_size, activation, input_shape);
    MaxPooling2D(pool_size);
    Dense(units, activation);
    Dropout(rate);
};
```

#### **Layer Types**:
- `Conv2D(filters, kernel_size, activation, input_shape)`
- `Dense(units, activation)`
- `MaxPooling2D(pool_size)`
- `Dropout(rate)`
- `Flatten()`
- `LSTM(units)`
- `BatchNormalization()`

### **2. Dataset Handling**

You can define datasets, apply augmentations, and split them into training, validation, and test sets. For example:

```aim
dataset "data/cifar10" split(train=70%, val=15%, test=15%) augment rotate flip normalize;
```

- **File Path**: The dataset path (e.g., CSV or image directory).
- **Split**: Dataset split into training, validation, and test.
- **Augmentations**: Augmentations like `rotate`, `flip`, etc.
- **Normalize**: Normalize data to standardize inputs.

### **3. Training Configuration**

The `train` keyword is used to configure the training process:

```aim
train ModelName on "data/dataset" for epochs using Optimizer(learning_rate=value) batch_size=value;
```

You can configure the following options:
- **Optimizer**: Choose an optimizer like Adam, SGD, or RMSProp.
- **Learning Rate**: Define the learning rate for the optimizer.
- **Batch Size**: Specify the batch size for training.
- **Epochs**: Set the number of training epochs.

### **4. Hyperparameter Tuning**

Traino supports **Optuna** for hyperparameter optimization. You can tune learning rate, batch size, and other parameters:

```aim
hyperparameter_tune ModelName using RandomSearch(lr=[0.001, 0.01, 0.1], batch_size=[32, 64, 128]) for 50 trials;
```

- **RandomSearch**: A random search algorithm for hyperparameter optimization.
- **GridSearch**: A grid search algorithm can also be implemented for exhaustive search.

### **5. Model Evaluation**

Evaluate the model using multiple metrics such as accuracy, precision, recall, F1-score, and AUC-ROC:

```aim
evaluate ModelName using "data/test_dataset" metrics=accuracy precision recall f1_score auc_roc;
```

### **6. Saving the Model**

After training, save the trained model in the desired format:

```aim
save ModelName as "model_filename.pt";
```

You can save the model in **PyTorch (.pt)** or **ONNX (.onnx)** formats.

---

## **Advanced Features**

### **Multi-GPU Training**

If you have multiple GPUs available, Traino will automatically use them to speed up training. No additional configuration is required.

### **Model Ensembling**

You can combine multiple models to form an ensemble using techniques like bagging, boosting, or stacking.

### **Pre-trained Model Support**

Traino supports using pre-trained models (e.g., **ResNet**, **VGG**) for transfer learning. You can fine-tune these models on your custom dataset:

```aim
model ResNet18 {
    ResNet18();
};

train ResNet18 on "data/cifar10" for 50 epochs using Adam(learning_rate=0.001) batch_size=64;

save ResNet18 as "resnet18_model.pt";
```

### **Text Preprocessing for NLP**

For Natural Language Processing (NLP) tasks, Traino provides preprocessing options such as tokenization, stopword removal, and lemmatization:

```aim
dataset "data/text_data.csv" split(train=70%, val=15%, test=15%) preprocess tokenization stopword_removal lemmatization;
```

---

## **Examples**

### **Basic Model Definition and Training**

```aim
model SimpleNN {
    Dense(128, activation="relu");
    Dense(10, activation="softmax");
};

dataset "data/cifar10" split(train=70%, val=15%, test=15%) normalize;

train SimpleNN on "data/cifar10" for 50 epochs using Adam(learning_rate=0.001) batch_size=64;

save SimpleNN as "simple_nn_model.pt";
```

### **Model with Hyperparameter Tuning**

```aim
model SimpleNN {
    Dense(128, activation="relu");
    Dense(10, activation="softmax");
};

dataset "data/cifar10" split(train=70%, val=15%, test=15%) normalize;

hyperparameter_tune SimpleNN using RandomSearch(lr=[0.001, 0.01], batch_size=[32, 64]) for 50 trials;

train SimpleNN on "data/cifar10" for 50 epochs using Adam(learning_rate=0.001) batch_size=64;

save SimpleNN as "simple_nn_model.pt";
```

---

## **TensorBoard Integration**

To enable TensorBoard logging during training, set `log_metrics=True`:

```aim
train SimpleNN on "data/cifar10" for 50 epochs using Adam(learning_rate=0.001) batch_size=64 log_metrics=True;
```

This will log training metrics (loss and accuracy) to TensorBoard.

---

## **Conclusion**

**Traino** is a powerful, easy-to-use AI programming language that simplifies model building, training, and evaluation. It abstracts away the complexities of machine learning frameworks, allowing you to focus on model design and training. With support for hyperparameter tuning, multi-GPU training, model ensembling, pre-trained models, and logging, Traino provides a comprehensive solution for AI model development.

Traino is designed for both beginners and experienced practitioners, making it an ideal tool for quickly experimenting with AI models and deploying them in production environments.

---

### **Contributing**

If you’d like to contribute to Traino, please fork the repository and submit pull requests with your improvements. If you have any questions or feedback, feel free to open an issue.

---

### **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
