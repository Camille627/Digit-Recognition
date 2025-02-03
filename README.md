# Project: Handwritten Digit Recognition with a Neural Network

## 📌 Description
This project implements a neural network to recognize handwritten digits using the MNIST dataset. It uses Python and NumPy to train a simple model without deep learning frameworks.

## 📂 Project Contents
- `digit_recognition.py`: Main script that loads data, trains the network, and tests accuracy.
- `README.md`: Project documentation.
- `accuracy_plot.png`: Graph illustrating accuracy evolution (generated after training).

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
- **Optimization** using gradient descent and weight updates.

## 🚀 Running the Project
1. Clone the project:
   ```bash
   git clone https://github.com/your-repo/mnist-digit-recognition.git
   cd mnist-digit-recognition
   ```
2. Run the Python script:
   ```bash
   python digit_recognition.py
   ```
3. Once training is complete, an image *accuracy_plot.png* is generated.

## 📊 Results
Here is the accuracy evolution during training:

![Accuracy Curve](accuracy_plot.png)

## 📜 References
- MNIST Dataset: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)
- Model Inspiration: [YouTube Video](https://www.youtube.com/watch?v=w8yWXqWQYmU)

## 🛠 Technologies Used
- Python
- NumPy
- Matplotlib

---
👨‍💻 **Author: Camille ANSEL**  
📅 **Date: 03/02/2025**

