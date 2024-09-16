# Created by Yifei Zhang

import numpy as np
import random
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import os

# File imports
from lstm_preprocessing import preprocess, preprocessing_one_trial_out
from lstm_model import evaluate_model, evaluate_model_ec, evaluate_model_mamba, run_genetic_algorithm, Individual

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8192)])
  except RuntimeError as e:
    print("Error: ", e)

from tensorflow.keras.mixed_precision import experimental as mixed_precision

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)
tf.debugging.set_log_device_placement(True)

REPEATS = 10
random.seed(41)
GESTURE_MAPPING = {
    0: 'G1',
    1: 'G2',
    2: 'G3',
    3: 'G4',
    4: 'G5',
    5: 'G6',
    6: 'G8',
    7: 'G9',
    8: 'G10',
    9: 'G11',
}

if __name__ == '__main__':

    # train_data, validation_data, train_labels, validation_labels, gesture_counts = preprocess()
    train_data, validation_data, test_data, train_labels, validation_labels, test_labels, train_gesture_counts, test_gesture_counts = preprocessing_one_trial_out()
    # print("Gesture counts in the dataset:")
    # # for gesture, count in sorted(gesture_counts.items()):
    # #     print(f"Gesture {gesture}: {count} occurrences")
    # display_gesture_counts(train_gesture_counts, GESTURE_MAPPING)
    print(train_data.shape)
    print(train_labels.shape)
    print(validation_data.shape)
    print(validation_labels.shape)

    # Repeat experiment for no EC (Random benchmark)
    scores = list()
    save_model, save_history= 0, 0
    for r in range(REPEATS):
        no_ec_history, score, no_ec_model = evaluate_model(train_data, train_labels, validation_data, validation_labels)
        score = score * 100.0
        print('>#%d: %.3f' % (r+1, score))
        scores.append(score)
        if score > np.argmax(scores):
            save_model = no_ec_model
            save_history = no_ec_history
    m_acc_no_ec, s = np.mean(scores), np.std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m_acc_no_ec, s))
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'JIGSAWS_LSTM_no_ec.h5')
    no_ec_model.save(filename)  # Save no EC model

    # Save no EC training history
    no_ec_history_dict = {
        "loss": no_ec_history.history['loss'],
        "val_loss": no_ec_history.history.get('val_loss', []),
        "accuracy": no_ec_history.history['accuracy'],
        "val_accuracy": no_ec_history.history['val_accuracy'],
    }
    filename = os.path.join(dirname, 'no_ec_training_history.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(no_ec_history_dict, f)

    # Genetic algorithm for hyperparameter tuning
    best_hyperparameters, generation_accuracy, best_hyperparameters_per_generation = run_genetic_algorithm(train_data, train_labels, validation_data, validation_labels, population_size=20, generations=50, tournament_size=3)
    # generation_accuracy = [x * 100 for x in generation_accuracy]
    print(generation_accuracy)

    # best_hyperparameters = Individual()
    # generation_accuracy = [0,0]
    # scores = [0,0]

    # Retrain model
    ec_history, lstm_ec_accuracy, ec_model = evaluate_model_ec(train_data, train_labels, validation_data, validation_labels, 
                                    best_hyperparameters.lstm_units, best_hyperparameters.dropout_rate, 
                                    best_hyperparameters.dense_units, best_hyperparameters.epochs, 
                                    best_hyperparameters.batch_size, best_hyperparameters.hidden_layer)

    print(f"Final model accuracy with optimized hyperparameters: {lstm_ec_accuracy * 100:.2f}%")
    filename = os.path.join(dirname, 'JIGSAWS_LSTM_ec.h5')
    ec_model.save(filename, save_format="tf")    # Save EC model
    # filename = os.path.join(dirname, 'ec_mamba_weights')
    # ec_model.save_weights(filename)


    # Save EC training history
    ec_history_dict = {
        "loss": ec_history.history['loss'],
        "val_loss": ec_history.history.get('val_loss', []),
        "accuracy": ec_history.history['accuracy'],
        "val_accuracy": ec_history.history['val_accuracy'],
    }
    filename = os.path.join(dirname, 'ec_training_history.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(ec_history_dict, f)

    # # Plotting the comparison
    # plt.figure(figsize=(10, 6))
    # plt.plot(scores, label='LSTM without EC', marker='o')
    # plt.plot(generation_accuracy, label='LSTM with EC', marker='x')
    # plt.xlabel('Experiment Iteration')
    # plt.ylabel('Accuracy')
    # plt.title('Comparison of LSTM Accuracies With and Without Evolutionary Computing (EC)')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    print(f"Best Hyperparameters: LSTM Units: {best_hyperparameters.lstm_units}, "
          f"Dropout: {best_hyperparameters.dropout_rate}, "
          f"Dense Units: {best_hyperparameters.dense_units}, "
          f"Epochs: {best_hyperparameters.epochs}, "
          f"Batch Size: {best_hyperparameters.batch_size}, "
          f"Hidden Layer: {best_hyperparameters.hidden_layer}")
    
    # Extract hyperparameter values
    lstm_units = [individual.lstm_units for individual in best_hyperparameters_per_generation]
    dropout_rates = [individual.dropout_rate for individual in best_hyperparameters_per_generation]

    generations = range(1, len(best_hyperparameters_per_generation) + 1)

    # Plot LSTM Units Evolution
    plt.figure(figsize=(14, 7))
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
    plt.plot(generations, lstm_units, marker='o', linestyle='-')
    plt.title('Evolution of LSTM Units')
    plt.xlabel('Generation')
    plt.ylabel('LSTM Units')

    # Plot Dropout Rates Evolution
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
    plt.plot(generations, dropout_rates, marker='o', linestyle='-', color='orange')
    plt.title('Evolution of Dropout Rate')
    plt.xlabel('Generation')
    plt.ylabel('Dropout Rate')

    plt.tight_layout()
    plt.show()

    # Population's Fitness Evolution
    average_fitness_per_generation, best_fitness_per_generation = zip(*generation_accuracy)
    generations = range(1, len(generation_accuracy) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(generations, average_fitness_per_generation, label='Average Fitness', marker='o')
    plt.plot(generations, best_fitness_per_generation, label='Best Fitness', marker='x', linestyle='--')
    plt.title('Evolution of Population Fitness Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True)
    plt.show()
