# Binary Classifier GUI

This project is the second in my artificial intelligence course and demonstrates the implementation of a logistic regression binary classifier with a graphical user interface (GUI). The GUI allows users to input learning parameters, train the classifier, and visualize the decision boundary.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Project Structure](#project-structure)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Overview
This project uses Tkinter for the GUI and NumPy and Matplotlib for the classifier and visualization. Users can input the learning rate and maximum number of iterations to train the logistic regression model on a dummy dataset. The decision boundary and data points are then visualized.

## Features
- User-friendly GUI to input training parameters.
- Logistic regression implementation with gradient descent.
- Visualization of decision boundary using Matplotlib.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/binary-classifier-gui.git
    cd binary-classifier-gui
    ```
2. Install the required dependencies:
    ```bash
    pip install numpy matplotlib
    ```

## Usage
1. Run the application:
    ```bash
    python main.py
    ```
2. Use the GUI to:
   - Input the learning rate.
   - Input the maximum number of iterations.
   - Click the "Train" button to train the classifier and visualize the decision boundary.

## Dependencies
- Python 3.12.2
- NumPy
- Matplotlib
- Tkinter (comes with Python standard library)


## Acknowledgements
This project is part of my artificial intelligence course. Special thanks to my instructor Dr.Emad Natsheh for his guidance and support.

