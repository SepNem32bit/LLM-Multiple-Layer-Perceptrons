# MLP From Scratch: Character Prediction Model
This project implements a simple Multi-Layer Perceptron (MLP) from scratch using PyTorch. The model is designed to predict the next character in a sequence of characters, trained on a dataset of names. The training data is loaded from a file called names.txt, where each line represents a name.

## Features  
**MLP Architecture**: The neural network model consists of:  
- **An embedding layer** to represent each character in a fixed-dimensional space.  
- **Two fully connected layers** with activation functions:  
  - **Tanh** activation in the first layer to introduce non-linearity.  
  - **Softmax** activation in the output layer for probability distribution.  
- **Cross-entropy loss** for training the model efficiently.  

**Training**: The model uses mini-batch gradient descent and updates weights after each epoch.   

**Validation**: The model evaluates its performance on a validation dataset after training.  


## File Structure
```bash
project_directory/
│
├── main.py                # Main entry point for training and testing
├── MLP_Training.py        # MLP Training class and model architecture
├── names.txt              # Dataset of names used for training
├── README.md              # Project documentation

```
## Dependencies
To run this project, you will need the following dependencies:

Python 3.x
PyTorch
You can install PyTorch using the following command:

```bash
pip install torch
```

## Getting Started
### 1. Prepare Dataset
The dataset (names.txt) should be a text file where each line represents a name. Here's an example:

```txt
Alice
Bob
Charlie
David
Eve
...
```
### 2. Run the Code
To train the model, run the following command in your terminal:

```bash
python main.py
```
This will:
- Load the dataset from `names.txt`.
- Split the data into training, validation, and test sets.
- Train the MLP model on the dataset.
- Print the loss after each training epoch and validation loss.