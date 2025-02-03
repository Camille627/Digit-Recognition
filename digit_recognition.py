### Author: Camille ANSEL 
### Date:   03/02/2025

#Inspiré de https://www.youtube.com/watch?v=w8yWXqWQYmU

import struct
import numpy as np
import matplotlib.pyplot as plt
import os

### Mise en forme des datasets
## Recuperation

def load_mnist_images(filename):
    """Charge les images MNIST depuis un fichier IDX3-UBYTE."""
    with open(filename, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))  # Lire l'en-tête
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)  # Charger et reformater
    return images

def load_mnist_labels(filename):
    """Charge les labels MNIST depuis un fichier IDX1-UBYTE."""
    with open(filename, 'rb') as f:
        magic, num_labels = struct.unpack(">II", f.read(8))  # Lire l'en-tête
        labels = np.frombuffer(f.read(), dtype=np.uint8)  # Charger les labels
    return labels

# 📍 Modifier le chemin de base vers ton dossier MNIST
base_path = r"C:\Users\Camille\Documents\01-Documents\04-projets\02-Programmation\Python\IA\MNIST\Datasets"

# Charger les fichiers
train_images = load_mnist_images(os.path.join(base_path, "train-images.idx3-ubyte"))
train_labels = load_mnist_labels(os.path.join(base_path, "train-labels.idx1-ubyte"))
test_images = load_mnist_images(os.path.join(base_path, "t10k-images.idx3-ubyte"))
test_labels = load_mnist_labels(os.path.join(base_path, "t10k-labels.idx1-ubyte"))

# Vérifier les dimensions
print(f"Train images: {train_images.shape}, Train labels: {train_labels.shape}")
print(f"Test images: {test_images.shape}, Test labels: {test_labels.shape}")

# Afficher un exemple
plt.imshow(train_images[0], cmap='gray')
plt.title(f"Exemple de chiffre : {train_labels[0]}")
plt.show()

## Transformation des images 28x28 en vecteurs de 784 éléments
print("-----------")
train_images = train_images.reshape(train_images.shape[0], 784).T  # Transpose pour que chaque colonne soit une image
test_images = test_images.reshape(test_images.shape[0], 784).T


# Vérification des nouvelles dimensions
print(f"Nouvelles dimensions des images d'entraînement : {train_images.shape}")
print(f"Nouvelles dimensions des images de test : {test_images.shape}")


## Normalisation des images (valeurs entre 0 et 1)
print("-----------")
train_images = train_images.astype(np.float32) / 255.0
test_images = test_images.astype(np.float32) / 255.0

# Vérification des nouvelles valeurs
print(f"Valeurs min/max après normalisation: {train_images.min()} - {train_images.max()}")
print("-----------")

def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

### Activation functions
# for the hidden layer
def activation_func(Z):
    # improved ReLU function (Leaky ReLU)
    return np.maximum(0.01*Z,Z)

# the activation function of the output layer
# adapted when their are several choices
def softmax(Z):
    return np.exp(Z) / sum(np.exp(Z))


def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = activation_func(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

### Derivative of activation function
def deriv_activ_func(Z):
    return np.where(Z > 0, 1, 0.01)

### Conversion de Y en la sortie attendtue de l'output layer
def one_hot(Y):
    # Y peut être une array
    one_hot_Y = np.zeros((Y.size,Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    m = X.shape[1]
    Y = one_hot(Y)
    dZ2 = A2 - Y
    dW2 = (1/m) * dZ2.dot(A1.T)
    db2 = (1/m) * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * deriv_activ_func(Z1)
    dW1 = (1/m) * dZ1.dot(X.T)
    db1 = (1/m) * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

### Function for results analyse
def prediction(A2):
    # return the value of the predicted number
    return np.argmax(A2,0)

def accuracy(predictions, Y):
    # print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1,A1,Z2,A2,W2,X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if (i % 50 == 0):
            print("iteration:",i," accuracy:",accuracy(prediction(A2),Y))
    return W1, b1, W2, b2

W1, b1, W2, b2 = gradient_descent(train_images, train_labels, 500, 0.10)