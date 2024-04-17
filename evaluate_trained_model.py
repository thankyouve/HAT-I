from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
from lstm_preprocessing import preprocess, preprocessing_one_trial_out, display_gesture_counts
from lstm_model import MAMBA_Model, evaluate_model_mamba
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
import os
import time
import editdistance
from tensorflow.keras.utils import plot_model


# Set seed
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

def plot_gesture_counts(gesture_labels, gesture_counts):
    """
    Plots a bar chart of gesture counts from a NumPy array.
    
    Parameters:
    - gesture_labels: A NumPy array or list of gesture labels (strings).
    - gesture_counts: A NumPy array of the counts for each gesture.
    """
    # To make sure gesture_labels and gesture_counts are NumPy arrays
    gesture_labels = np.array(gesture_labels)
    gesture_counts = np.array(gesture_counts)
    
    # Creating the bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(gesture_labels, gesture_counts, color='skyblue')
    
    plt.xlabel('Gesture Label')
    plt.ylabel('Count')
    plt.title('Gesture Counts in the Dataset')
    plt.xticks(rotation=45)  # Rotate labels to prevent overlap
    plt.grid(axis='y')
    plt.show()

def pred_time(model, test, runs=10):
    """
    Measures the average prediction time of a model.

    Parameters:
    - model: The model to measure prediction time for.
    - test: The dataset to predict on.
    - runs: The number of times to run the prediction for averaging.

    Returns:
    - The average prediction time in seconds.
    """
    # Warm-up model
    model.predict(test[:1])
    
    start_time = time.time()
    for _ in range(runs):
        model.predict(test)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / runs
    return avg_time

def edit_dist(predictions, ground_truth):
    """
    Computes the edit distance between model predictions and ground truth labels
    for 1-dimensional arrays.

    Parameters:
    - predictions: 1-dimensional array of model predictions.
    - ground_truth: 1-dimensional array of actual ground truth labels.

    Returns:
    - The average edit distance to the ground truth.
    """
    total_distance = 0
    
    # To make sure predictions and ground_truth are numpy arrays
    predictions = np.array(predictions, dtype=str)  # Convert to string
    ground_truth = np.array(ground_truth, dtype=str)

    # Calculate edit distance
    for pred, true in zip(predictions, ground_truth):
        distance = editdistance.eval([pred], [true])
        total_distance += distance
    
    average_distance = total_distance / len(predictions)
    return average_distance

# Load the model and data
train_data, validation_data, test_data, train_labels, validation_labels, test_labels, train_gesture_counts, test_gesture_counts = preprocessing_one_trial_out()
# train_data, test_data, train_labels, test_labels, gesture_counts = preprocess()
loaded_model = load_model('E:/School/McmasterU/HAT&I/Model/jigsaws_lstm/JIGSAWS_LSTM_ec.h5')

print("testlabels:")
print(test_labels.shape)
# loaded_model = MAMBA_Model(50,150,10,0.2797845958094579)
# loaded_model.load_weights('E:/School/McmasterU/HAT&I/Model/jigsaws_lstm/ec_mamba_weights')
print("Gesture counts in the training set:")
display_gesture_counts(train_gesture_counts, GESTURE_MAPPING)
print("Gesture counts in the testing set:")
display_gesture_counts(test_gesture_counts, GESTURE_MAPPING)

gesture_labels = np.array(['Gesture 1', 'Gesture 2', 'Gesture 3', 'Gesture 4', 'Gesture 5', 'Gesture 6', 'Gesture 8', 'Gesture 9', 'Gesture 10', 'Gesture 11'])

plot_gesture_counts(gesture_labels, train_gesture_counts)
plot_gesture_counts(gesture_labels, test_gesture_counts)

# Plot loaded model architecture
plot_model(loaded_model, to_file='model_diagram.png', show_shapes=True, show_layer_names=True)

# Generate predicted and true labels
predicted_labels = loaded_model.predict(test_data)
predicted_labels = np.argmax(predicted_labels, axis=-1)
true_labels = np.argmax(test_labels, axis=-1)
print(predicted_labels)
print(true_labels)

# Flatten prediction and labels as they are independent timesteps
flattened_predictions = predicted_labels.flatten()
flattened_true_labels = true_labels.flatten()

# # Loss and accuracy
# loaded_model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=['accuracy'])
# loss, accuracy = loaded_model.evaluate(test_data, test_labels)
# print("Loss:", loss)
# print("Accuracy:", accuracy)

# Display confusion matrix
cm = confusion_matrix(flattened_true_labels, flattened_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# Loss curve
with open('E:/School/McmasterU/HAT&I/Model/jigsaws_lstm/ec_training_history.pkl', 'rb') as f:
    history = pickle.load(f)

training_losses = history['loss']
validation_losses = history['val_loss']

plt.plot(training_losses, label='Training Loss')
plt.plot(validation_losses, label='Validation Loss')
plt.title('Training vs. Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Size comparison
size_model1 = os.path.getsize('E:/School/McmasterU/HAT&I/Model/jigsaws_lstm/JIGSAWS_LSTM_no_ec.h5')
size_model2 = os.path.getsize('E:/School/McmasterU/HAT&I/Model/jigsaws_lstm/JIGSAWS_LSTM_ec.h5')

print(f"Size of Model 1: {size_model1 / (1024**2):.2f} MB")
print(f"Size of Model 2: {size_model2 / (1024**2):.2f} MB")

model1 = load_model('E:/School/McmasterU/HAT&I/Model/jigsaws_lstm/JIGSAWS_LSTM_no_ec.h5')
model2 = load_model('E:/School/McmasterU/HAT&I/Model/jigsaws_lstm/JIGSAWS_LSTM_ec.h5')
model1.summary()
model2.summary()

# Time comparison
avg_time_model1 = pred_time(model1, test_data)
avg_time_model2 = pred_time(model2, test_data)

print(f"Average Prediction Time for Model 1: {avg_time_model1:.6f} seconds")
print(f"Average Prediction Time for Model 2: {avg_time_model2:.6f} seconds")

# Edit distance
predictions_model1 = model1.predict(test_data)
predictions_model2 = model2.predict(test_data)

predictions_model1 = np.argmax(predictions_model1, axis=-1)
predictions_model2 = np.argmax(predictions_model2, axis=-1)

avg_dist_model1 = edit_dist(predictions_model1, true_labels)
avg_dist_model2 = edit_dist(predictions_model2, true_labels)

print(f"Average Edit Distance to Ground Truth for Model 1: {avg_dist_model1}")
print(f"Average Edit Distance to Ground Truth for Model 2: {avg_dist_model2}")