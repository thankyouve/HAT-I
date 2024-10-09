# Surgical Gesture Recognition using LSTM and BiLSTM with Evolutionary Hyperparameter Tuning

This project implements Long Short-Term Memory (LSTM) and Bi-directional LSTM (BiLSTM) models for surgical gesture recognition on the JIGSAWS dataset, with the option for evolutionary hyperparameter tuning to improve model performance.

## Project Structure

- `main.py`: Entry point for running the model.
- `lstm_model.py`: Contains functions for building LSTM and BiLSTM models.
- `lstm_preprocessing.py`: Preprocessing JIGSAWS kinematic data into formats compatible with the models.
- `evaluate_trained_model.py`: Functions for loading and evaluating the trained models.
- `Experimental_setup/`: Contains training/testing splits for one-trial-out experiments.

## Features

- **LSTM & BiLSTM Models**: Choose between standard LSTM and BiLSTM models for time-series data.
- **Evolutionary Computation**: Hyperparameter optimization using evolutionary algorithms.
- **Sequence-to-Sequence Classification**: Predicts gesture sequences from kinematic data.

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
2. **Preprocess Data**: Ensure the JIGSAWS dataset is available and preprocess it using lstm_preprocessing.py.
3. **Run Model**: Modify and execute main.py for training and evaluating the model.

## Usage

**Training a BiLSTM Model**
Modify main.py to call the BiLSTM function and set your desired parameters.

**Hyperparameter Tuning with Evolutionary Computation**
To perform evolutionary hyperparameter tuning, use the evaluate_model_ec method within lstm_model.py.

**Data Preprocessing**
Use lstm_preprocessing.py to format the kinematic data and gesture labels into the appropriate 3D numpy arrays for training and evaluation.

## Future Work
Implement real-time gesture recognition.
Add additional datasets for more comprehensive model validation.
Optimize evolutionary computation for larger hyperparameter spaces.
