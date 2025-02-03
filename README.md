# Project: Handwritten Digit Recognition with a Neural Network

## 📌 Description
This project implements a neural network to recognize handwritten digits using the MNIST dataset. It uses Python and NumPy to train a simple model without deep learning frameworks.

## 📂 Project Contents
- `digit_recognition.py`: Main script that loads data, trains the network, and tests accuracy.
- `accuracy_plot.png`: Screenshot of the console illustrating accuracy evolution (generated after training).

## 📥 Data Preparation
The MNIST dataset is used for training. The images are:
- Loaded in IDX3-UBYTE format.
- Normalized between 0 and 1.
- Converted into 784-element vectors.
- Labels are converted to one-hot encoding.

## 🧠 Model Architecture
The neural network consists of:
- **A hidden layer** with 10 neurons and *Leaky ReLU* activation.
- **An output layer** with 10 neurons and *softmax* activation.

## 📊 Results
Here is the accuracy evolution during training:

![Console screenshot that show an accuracy of 84% after 500 gradient descent](Results/digit_reco_v1_results.png)

## 📜 References
- MNIST Dataset: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)
- Model Inspiration: [YouTube Video](https://www.youtube.com/watch?v=w8yWXqWQYmU)

## 🛠 Package Used
- Python
- NumPy
- Matplotlib

---
👨‍💻 **Author: Camille ANSEL**  
📅 **Date: 03/02/2025**

