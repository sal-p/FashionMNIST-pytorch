# FashionMNIST Image Classifier with pytorch

Model building, training, and testing using pytorch on FashionMNIST images.

This project contains a simple implementation of a neural network using PyTorch for image classification. The model is trained and tested on the FashionMNIST dataset, a dataset consisting of 70,000 grayscale images belonging to 10 different categories (e.g., T-shirt, trousers, pullover, dress, etc.).

## Requirements

- PyTorch
- torchvision

## Files

- `main.py`: This is the main script file that contains all the necessary code to train and test the model.

## Usage

You can simply run the `main.py` script to train and evaluate the neural network:

```bash
python main.py
```

## Implementation Details

The main script includes the following stages:

- **Data Loading:** The FashionMNIST dataset is loaded from torchvision, transformed into tensors and split into training and testing data. A dataloader object is created to enable easy iteration and batching of the dataset.

- **Model Definition:** A simple feed-forward neural network model is defined with two hidden layers, each of size 512. The input size is 784 (since FashionMNIST images are 28x28), and the output size is 10, corresponding to the 10 classes in the dataset.

- **Training Loop:** For each epoch, the model makes predictions on the input, calculates the loss, performs backpropagation to adjust the weights, and updates the optimizer.

- **Testing Loop:** After each epoch, the model is evaluated on the test data. The function calculates the average loss and accuracy of the model on the test set.

- **Model Evaluation:** The model's performance is evaluated at the end of each epoch and printed on the console. 

## Hyperparameters

The model uses Stochastic Gradient Descent (SGD) for optimization with a learning rate of `1e-3`. The batch size for the dataloader is set to 64, and the model is trained for a total of 10 epochs.

Please note that the current configuration and hyperparameters are basic and may not result in the best possible performance of the model. You are encouraged to experiment with different configurations to get the best results.
