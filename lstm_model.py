# Created by Yifei Zhang
import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Bidirectional, Layer, MultiHeadAttention, LayerNormalization, Flatten
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tqdm.keras import TqdmCallback
import tensorflow as tf

# # Uncomment for multiprocessing
# import multiprocessing

# Evaluate Model without EC
def evaluate_model(train_data, train_labels, test_data, test_labels):
    verbose, epochs, batch_size = 0, 40, 64
    timesteps, features, outputs = train_data.shape[1], train_data.shape[2], train_labels.shape[1]
    model = Sequential()
    model.add(Bidirectional(LSTM(1024, input_shape=(timesteps, features), return_sequences=False)))
    model.add(Dropout(0.5))
    # model.add(TimeDistributed(Dense(100, activation='relu')))
    # model.add(TimeDistributed(Dense(outputs, activation='softmax')))
    # model.add(Dense(100, activation='relu'))
    model.add(Dense(outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_data=(test_data, test_labels), callbacks=[TqdmCallback(verbose=0)]) # callbacks=[TqdmCallback(verbose=0)]
    _, accuracy = model.evaluate(test_data, test_labels, batch_size=batch_size, verbose=2)
    # print(test_data[0:1])
    # print(model.predict(test_data[0:1]))
    # print(test_labels[0])
    return history, accuracy, model

# Modify the evaluate_model function to accept hyperparameters (Evaluate model with EC)
def evaluate_model_ec(train_data, train_labels, test_data, test_labels, lstm_units, dropout_rate, dense_units, epochs, batch_size, hidden_layer):
    timesteps, features, outputs = train_data.shape[1], train_data.shape[2], train_labels.shape[1]
    model = Sequential()
    if hidden_layer == 1:
        model.add(Bidirectional(LSTM(lstm_units, input_shape=(timesteps, features), return_sequences=False)))
        model.add(Dropout(dropout_rate))
    else:
        model.add(Bidirectional(LSTM(lstm_units, input_shape=(timesteps, features), return_sequences=True)))
        model.add(Dropout(dropout_rate))
        model.add(Bidirectional(LSTM(lstm_units, return_sequences=False)))
        model.add(Dropout(dropout_rate))
    # model.add(TimeDistributed(Dense(dense_units, activation='relu')))
    # model.add(TimeDistributed(Dense(outputs, activation='softmax')))
    # model.add(Dense(dense_units, activation='relu'))
    model.add(Dense(outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='accuracy',  # Monitor training accuracy
                                   patience=3,  # Number of epochs with no improvement after which training will be stopped
                                   mode='max',  # The direction is "maximize" for accuracy
                                   verbose=1)
    history = model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, verbose=0, validation_data=(test_data,test_labels), callbacks=[TqdmCallback(verbose=0), early_stopping]) # callbacks=[TqdmCallback(verbose=0)]
    _, accuracy = model.evaluate(test_data, test_labels, batch_size=batch_size, verbose=2)
    # print(model.predict(test_data))
    return history, accuracy, model

# Genetic algorithm setup
class Individual:
    def __init__(self):
        # self.lstm_units = random.choice(range(50, 301, 50))
        self.hidden_layer = random.choice([1, 2])
        self.lstm_units = random.choice([64, 128, 256, 512, 1024])
        self.dropout_rate = random.uniform(0.1, 0.5)
        self.dense_units = random.choice(range(50, 201, 50))
        self.epochs = random.choice(range(10, 51, 10))
        self.batch_size = random.choice([64, 128, 256])
        # self.batch_size = random.choice(range(5, 21, 5))
        self.fitness_score = -1
        self.model = -1

    def fitness(self, train_data, train_labels, test_data, test_labels):
        _, accuracy, model = evaluate_model_ec(train_data, train_labels, test_data, test_labels, 
                              self.lstm_units, self.dropout_rate, self.dense_units, 
                              self.epochs, self.batch_size, self.hidden_layer)
        # _, accuracy, model = evaluate_model_mamba(train_data, train_labels, test_data, test_labels, 
        #                       self.lstm_units, self.dropout_rate, self.dense_units, 
        #                       self.epochs, self.batch_size)        
        return accuracy, model
    
    def display(self):
        print(f"hidden_layer: {self.hidden_layer}")
        print(f"lstm_units: {self.lstm_units}")
        print(f"dropout_rate: {self.dropout_rate}")
        print(f"dense_units: {self.dense_units}")
        print(f"epochs: {self.epochs}")
        print(f"batch_size: {self.batch_size}")

def crossover(parent1, parent2):
    child = Individual()
    child.hidden_layer = random.choice([parent1.hidden_layer, parent2.hidden_layer])
    child.lstm_units = random.choice([parent1.lstm_units, parent2.lstm_units])
    child.dropout_rate = np.mean([parent1.dropout_rate, parent2.dropout_rate])
    child.dense_units = random.choice([parent1.dense_units, parent2.dense_units])
    child.epochs = random.choice([parent1.epochs, parent2.epochs])
    child.batch_size = random.choice([parent1.batch_size, parent2.batch_size])
    return child

def mutate(individual):
    if random.random() < 0.1:
        individual.hidden_layer = random.choice([1, 2])
        # individual.lstm_units = random.choice(range(50, 301, 50))
        individual.lstm_units = random.choice([64, 128, 256, 512, 1024])
        individual.dropout_rate = random.uniform(0.1, 0.5)
        individual.dense_units = random.choice(range(50, 201, 50))
        individual.epochs = random.choice(range(10, 51, 10))
        # individual.batch_size = random.choice([64, 128, 256])
        individual.batch_size = random.choice(range(5, 21, 5))

def tournament_selection(population, tournament_size):
    tournament = random.sample(population, tournament_size)
    best_individual = max(tournament, key=lambda individual: individual.fitness_score)
    return best_individual

def initialize_population(size):
    return [Individual() for _ in range(size)]

def run_genetic_algorithm(train_data, train_labels, test_data, test_labels, population_size=10, generations=5, tournament_size=3):
    population = initialize_population(population_size)
    best_hyperparameters_per_generation = []  # To store the best hyperparameters at each generation
    generation_accuracies = []
    best_individual_overall = Individual()

    for generation in range(generations):
        # Evaluate fitness for each individual
        for individual in population:
            individual.display()
            individual.fitness_score, individual.model = individual.fitness(train_data, train_labels, test_data, test_labels)

        # Calculate and log average and best fitness in this generation
        average_fitness = np.mean([individual.fitness_score for individual in population])
        best_fitness = max([individual.fitness_score for individual in population])
        generation_accuracies.append((average_fitness, best_fitness))
        print(average_fitness)
        print(best_fitness)

        # Find and log the best individual's hyperparameters in this generation
        best_individual = max(population, key=lambda individual: individual.fitness_score)
        best_hyperparameters_per_generation.append(best_individual)
        if best_individual.fitness_score > best_individual_overall.fitness_score:
            best_individual_overall = best_individual
        print(best_individual_overall.fitness_score)

        # Tournament Selection for Crossover
        new_population = []
        while len(new_population) < population_size:
            parent1 = tournament_selection(population, tournament_size)
            parent2 = tournament_selection(population, tournament_size)
            child = crossover(parent1, parent2)
            new_population.append(child)

        # Mutation
        for individual in new_population:
            mutate(individual)

        population = new_population

    return best_individual_overall, generation_accuracies, best_hyperparameters_per_generation

# # For multiprocessing
# def evaluate_individual_fitness(individual, train_data, train_labels, test_data, test_labels):
#     return individual.fitness(train_data, train_labels, test_data, test_labels)

# def run_genetic_algorithm(train_data, train_labels, test_data, test_labels, population_size=10, generations=5, tournament_size=3):
#     population = initialize_population(population_size)
#     generation_accuracies = []  # List to store the best accuracy of each generation

#     for generation in range(generations):
#         print(f"Generation {generation + 1}")

#         # Parallelize the fitness evaluation using multiprocessing
#         with multiprocessing.Pool() as pool:
#             fitness_results = pool.starmap(evaluate_individual_fitness, [(individual, train_data, train_labels, test_data, test_labels) for individual in population])
        
#         # Assign fitness results to individuals
#         for individual, fitness_score in zip(population, fitness_results):
#             individual.fitness_score = fitness_score

#         # Find the best accuracy in this generation
#         best_accuracy_this_generation = max(individual.fitness_score for individual in population)
#         generation_accuracies.append(best_accuracy_this_generation)

#         # Tournament Selection for Crossover
#         new_population = []
#         while len(new_population) < population_size:
#             parent1 = tournament_selection(population, tournament_size)
#             parent2 = tournament_selection(population, tournament_size)
#             child = crossover(parent1, parent2)
#             new_population.append(child)

#         # Mutation
#         for individual in new_population:
#             mutate(individual)

#         population = new_population

#     # Find the best individual after all generations
#     best_individual = max(population, key=lambda ind: ind.fitness_score)
#     return best_individual, generation_accuracies

class MultiScaleAttention(Layer):
    def __init__(self, units, num_heads, dropout):
        super(MultiScaleAttention, self).__init__()
        # Multi-head attention for multiple scales
        self.multi_head_attention = MultiHeadAttention(num_heads=num_heads, key_dim=units)
        self.dropout = Dropout(0.1)
        self.layernorm = LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        # inputs would typically be the output of an LSTM or Bidirectional LSTM layer
        attn_output = self.multi_head_attention(inputs, inputs)
        attn_output = self.dropout(attn_output)
        out = self.layernorm(inputs + attn_output)
        return out

class MAMBA_Model(tf.keras.Model):
    def __init__(self, lstm_units, dense_units, num_classes, dropout):
        super(MAMBA_Model, self).__init__()
        self.bi_lstm = Bidirectional(LSTM(lstm_units, return_sequences=True))  # Adjusted to keep sequence dimension
        self.attention1 = MultiScaleAttention(lstm_units, num_heads=8, dropout=dropout)
        self.attention2 = MultiScaleAttention(lstm_units, num_heads=8, dropout=dropout)
        self.dense = Dense(dense_units, activation='relu')
        self.output_layer = Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.bi_lstm(inputs)
        x = self.attention1(x)
        x = self.attention2(x)
        # Flatten or pool the sequence dimension before the dense layer if needed
        x = Flatten()(x)  # Example: Flatten the output for the Dense layer, if your model ends with Dense layers for classification.
        x = self.dense(x)
        return self.output_layer(x)

def evaluate_model_mamba(train_data, train_labels, test_data, test_labels, lstm_units, dropout_rate, dense_units, epochs, batch_size):
    num_classes = 10
    model = MAMBA_Model(lstm_units, dense_units, num_classes, dropout_rate)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, verbose=0, validation_data=(test_data,test_labels)) # callbacks=[TqdmCallback(verbose=0)]
    _, accuracy = model.evaluate(test_data, test_labels, batch_size=batch_size, verbose=2)
    return history, accuracy, model