# Simple Pendulum Linear Regression

## Overview

This project demonstrates the application of linear regression to estimate the gravitational constant \( g \) using data from a simple pendulum experiment. The analysis involves loading data from an Excel file, visualizing it, and applying a linear regression model to interpret the results.

## Project Workflow

The script performs the following key steps:

1. **Data Loading**: Reads pendulum data from an Excel file.
2. **Data Preparation**: Organizes and processes the data for analysis.
3. **Data Visualization**: Creates visual representations of the data.
4. **Data Splitting**: Divides the data into training and testing subsets.
5. **Model Training**: Constructs and trains a linear regression model.
6. **Model Evaluation**: Assesses the performance of the model using various metrics.
7. **Gravitational Constant Estimation**: Calculates the gravitational constant \( g \) based on the model's parameters.
8. **Results Visualization**: Provides visual insights into the model’s predictions versus actual data.

## Requirements

To execute this script, ensure you have the following Python libraries installed:

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`

Install the necessary libraries using pip with the following command:

```bash
pip install numpy pandas matplotlib scikit-learn
