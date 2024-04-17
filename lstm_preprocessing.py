# Created by Yifei Zhang
# Use this file with main.py for LSTM training and prediction.
# This file specifies the preprocessing functions. 
# Change Path to root folder in preprocess() function before use, as well as what activity data to use. 
# Folder structure is assumed to be the same as JIGSAWS data downloaded. 

import numpy as np
import os
from tensorflow.keras.utils import to_categorical

# Specify what activity data to use
ACTIVITIES = ['Suturing']
SNIPPET_LEN = 30
DOWNSAMPLE_LEN = 6

GESTURE_MAPPING = {
    'G1': 0,
    'G2': 1,
    'G3': 2,
    'G4': 3,
    'G5': 4,
    'G6': 5,
    'G8': 6,
    'G9': 7,
    'G10': 8,
    'G11': 9,
}

def read_kinematic_files_from_folder(folder_path: str):
    '''
    Description:  Reads kinematic data from activity data folder(s) into a dictionary txt_files_content. 
    Inputs:       string (path to the kinematics folder)
    Outputs:      dictionary (dictionary of kinematics data of all file_names as key, and data as value)
    '''
    txt_files_content = {}
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            data = [list(map(float, line.strip().split())) for line in file]
            txt_files_content[filename] = np.array(data)
    return txt_files_content

def read_trial_file(file_path: str):
    '''
    Reads a single trial file and extracts gesture information.
    '''
    gestures = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            file_name, gesture = parts[0], parts[1]
            activity_name, participant_id, start_time, end_time = file_name.split('_')
            end_time = end_time.split('.')[0]
            gestures.append({
                'activity_name': activity_name,
                'participant_id': participant_id,
                'start_time': int(start_time),
                'end_time': int(end_time),
                'gesture': gesture
            })
    return gestures

def read_one_trial_out_folder(trial_folder: str):
    '''
    Reads the contents of a OneTrialOut folder, extracting both training and testing data.
    '''
    train_data = []
    test_data = []

    for filename in os.listdir(trial_folder):
        file_path = os.path.join(trial_folder, filename)
        if 'train' in filename.lower():
            train_data.extend(read_trial_file(file_path))
        elif 'test' in filename.lower():
            test_data.extend(read_trial_file(file_path))

    return train_data, test_data

# def extract_kinematic_data(data_folder: str, trial_data: dict):
#     '''
#     Extracts the kinematic data for the given trial data entries.
#     '''
#     kinematic_data = []
    
#     for entry in trial_data:
#         participant_id = entry['participant_id']
#         start_time = entry['start_time']
#         end_time = entry['end_time']
        
#         kinematic_file_path = os.path.join(data_folder, 'Suturing', 'kinematics', 'AllGestures', f"Suturing_{participant_id}.txt")
#         with open(kinematic_file_path, 'r') as file:
#             lines = file.readlines()
#             # Extract the lines corresponding to the start and end times
#             segment = lines[start_time:end_time+1]
#             kinematic_data.append({
#                 'gesture': entry['gesture'],
#                 'segment_data': segment
#             })

#     return kinematic_data

def extract_kinematic_data(data_folder: str, trial_data: dict):
    '''
    Extracts the kinematic data for the given trial data entries, focusing on specific columns,
    and converts the numeric data back into the original string format.
    '''
    kinematic_data = []
    
    # Specify the columns you want to keep
    # Note: Python uses 0-based indexing, so subtract 1 from each column number in your list
    columns_to_keep = [38, 39, 40, 50, 51, 52, 56, 57, 58, 59, 69, 70, 71, 75]
    
    for entry in trial_data:
        participant_id = entry['participant_id']
        start_time = entry['start_time']
        end_time = entry['end_time']
        
        kinematic_file_path = os.path.join(data_folder, 'Suturing', 'kinematics', 'AllGestures', f"Suturing_{participant_id}.txt")
        with open(kinematic_file_path, 'r') as file:
            # Load the entire file into a numpy array for easier slicing
            data = np.loadtxt(file, usecols=columns_to_keep)
            # Extract the segment of interest
            segment = data[start_time:end_time+1]
            # Convert each row in the segment back into a string format
            segment_data_as_strings = [" ".join(map(str, row)) for row in segment]
            kinematic_data.append({
                'gesture': GESTURE_MAPPING[entry['gesture']],
                'segment_data': segment_data_as_strings
            })

    return kinematic_data

def read_transcription_files_from_folder(folder_path: str):
    '''
    Description:  Reads transcription from activity data folder(s) into a dictionary txt_files_content. 
    Inputs:       string (path to the transcription folder)
    Outputs:      dictionary (dictionary of transcription of all file_names as key, and data as value)
    '''
    txt_files_content = {}
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            data = [list(map(str, line.strip().split())) for line in file]
            txt_files_content[filename] = np.array(data)
    return txt_files_content

def del_not_transcripted_kinematics(kinematics: dict, transcription:dict):
    '''
    Description:  Deletes the non-transcripted kinematic entries.
                  Some of the kinematics datas in activity folder(s) are not transcripted, which would be problematic
                  when processing data with label, as no true label can be generated. 
                  The function compares keys from the two dictionary inputs, and get rid of the one that does not exist in
                  dictionary of transcription. 
    Inputs:       dictionary (dicitonary of kinematics data), 
                  dictionary (dictionary of transcription)
    Outputs:      dictionary (dictionary of kinematics data that has same key entries with dictionary of transcription)
    '''
    modified_kinematics = {key: val for key, val in kinematics.items() if key in transcription.keys()}
    return modified_kinematics

def trim_kinematics_arr(kinematics: dict, transcription: dict):
    '''
    Description:  Trims kinematics array according to start time and end time specified in transcription. 
                  The function reads keys of dictionary of kinematics, and extract start time and end time of
                  the trial from the entry with same "filename" key in dictionary of transcription, then trims
                  kinematics data, deleting the part before start time and after end time. 
    Inputs:       dictionary (dicitonary of kinematics data), 
                  dictionary (dictionary of transcription)
    Outputs:      dictionary (dictionary of kinematics data that starts at start time and end at end time)
    '''
    for key_filename, kin_arr in kinematics.items():
        start_frame, end_frame = int(transcription[key_filename][0][0]), int(transcription[key_filename][-1][1])
        mask = np.ones(len(kin_arr), dtype = bool)
        mask[np.r_[0:start_frame, end_frame:len(mask)]] = False
        kinematics[key_filename] = kin_arr[mask]
    return kinematics

def generate_labels(transcription: dict):
    '''
    Description:  Generate labels according to dictionary of transcription.
                  The generated labels would be in a dictionary that has filenames as keys. The values would
                  be for each time step, from start time to end time, at the same length to each of the same 
                  name file in kinematics, such that the gestures would be time matching with kinematics data. 
                  Since we are using a LSTM, this function needs to have to_categorical, as LSTM only takes
                  categorical array as labels, but not like CNN can take different names as labels.
    Inputs:       dictionary (dictionary of transcription)
    Outputs:      dictionary (dictionary of labels)
                  dictionary (dictoinary of gestures)
    '''
    # label_dict = {}
    # for key, val in transcription.items():
    #     start_frame, end_frame = int(val[0][0]), int(val[-1][1])
    #     temp_arr = np.zeros((end_frame - start_frame, 1), dtype = int)
    #     for i in range(len(val)):
    #         temp_arr[int(val[i][0]) - start_frame : int(val[i][1]) - start_frame + 1] = int(val[i][2].strip('G'))
    #     label_dict[key] = to_categorical(temp_arr)
    # return label_dict
    label_dict = {}
    # gesture_counts = {}
    gesture_counts = np.zeros(12, dtype=int)

    for key, val in transcription.items():
        # This assumes that the third element in each line of the transcription is the gesture label
        gestures = [int(item[2].strip('G')) for item in val]
        for gesture in gestures:
            # gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1
            gesture_counts[gesture] += 1
        
        start_frame, end_frame = int(val[0][0]), int(val[-1][1])
        temp_arr = np.zeros((end_frame - start_frame, 1), dtype = int)
        for i in range(len(val)):
            gesture_label = int(val[i][2].strip('G'))
            temp_arr[int(val[i][0]) - start_frame : int(val[i][1]) - start_frame + 1] = gesture_label
        label_dict[key] = to_categorical(temp_arr)

    return label_dict, gesture_counts

def stack_dictionary_val(dict_arrays: dict):
    '''
    Description:  Stack all the dictionaries into a 3D numpy array. 
                  Since dictionaries cannot be used to train an LSTM, we change the format to 3D numpy array. 
                  Furthermore, as LSTM's nature, the time series needs to be at same length. Thus, we fill all
                  the data with NaN to the size of longest rows and columns, so they would be same size. We use
                  this function to stack both kinematics data and true labels, which should be the same size
                  across first two dimensions (entry_num and time_steps), and third dimension would be number of
                  features for the data, and total number of gestures for the labels. 
    Inputs:       dictionary (dictionary of kinematics/labels)
    Outputs:      numpy array (   1st dimension: entry_num, 
                                  2nd dimension: time_steps,
                                  3rd dimension: features/gestures)
    '''
    max_columns = max(arr.shape[1] for arr in dict_arrays.values())
    max_rows = max(arr.shape[0] for arr in dict_arrays.values())
    result = np.full((len(dict_arrays), max_rows, max_columns), np.nan)

    for i, key in enumerate(dict_arrays):
        arr = dict_arrays[key]
        result[i, :arr.shape[0], :arr.shape[1]] = arr
    return result

def convert_string_to_numeric_array(segment_strings: list):
    '''
    Convert a list of string representations of feature values into a numeric array.
    Each string in the input list represents one timestep, with features separated by spaces.
    '''
    return np.array([list(map(float, timestep.split())) for timestep in segment_strings])

def stack_segments(trial_data: dict):
    '''
    Description:  Stack all the dictionaries into a 3D numpy array. 
                  Since dictionaries cannot be used to train an LSTM, we change the format to 3D numpy array. 
                  Furthermore, as LSTM's nature, the time series needs to be at same length. Thus, we fill all
                  the data with NaN to the size of longest rows and columns, so they would be same size. We use
                  this function to stack both kinematics data and true labels, which should be the same size
                  across first two dimensions (entry_num and time_steps), and third dimension would be number of
                  features for the data, and total number of gestures for the labels. 
    Inputs:       dictionary (dictionary of kinematics/labels)
    Outputs:      numpy array (   1st dimension: entry_num, 
                                  2nd dimension: time_steps,
                                  3rd dimension: features/gestures)
    '''

    # Extract unique gestures and sort them to maintain consistency
    # unique_gestures = set(entry['gesture'] for entry in trial_data)
    # Create a mapping from gesture labels to integers
    num_classes = len(GESTURE_MAPPING)
    # print(num_classes)
    
    # Find the maximum length of the segments and number of features
    max_length = max(len(entry['segment_data']) for entry in trial_data)
    num_features = len(trial_data[0]['segment_data'][0].split())
    # num_features = trial_data[0]['segment_data'].shape[1]

    # Initialize arrays
    data_array = np.full((len(trial_data), max_length, num_features), np.nan, dtype=float)
    gesture_array = np.zeros((len(trial_data), max_length, num_classes), dtype=np.float32)
    
    for i, entry in enumerate(trial_data):
        # Convert segment data strings to numeric array
        segment_numeric = convert_string_to_numeric_array(entry['segment_data'])
        data_array[i, :len(segment_numeric), :] = segment_numeric
        
        # Map gesture label to integer and create a categorical encoding
        gesture_categorical = to_categorical([entry['gesture']] * len(segment_numeric), num_classes=num_classes)
        gesture_array[i, :len(gesture_categorical), :] = gesture_categorical
    
    return data_array, gesture_array

def split_into_snippets(data: list, labels: list, snippet_length: int):
    '''
    Description:  Splits data and labels into smaller time segments for training
                  As the time steps are usually long, we need to split the data into smaller snippets. The 
                  snippets are also checked to exclude any ones that has NaN in them, due to the previous
                  step to match sizes. 
    Inputs:       numpy array (kinematics data)
                  numpy array (true labels)
                  int (length of snippets)
    Outputs:      numpy array (   1st dimension: snippet_num, 
                                  2nd dimension: time_steps,
                                  3rd dimension: kinematics data)
                  numpy array (   1st dimension: snippet_num, 
                                  2nd dimension: time_steps,
                                  3rd dimension: labels)
    '''
    snippet_data = []
    snippet_labels = []

    for sample_data, sample_label in zip(data, labels):
        for i in range(0, len(sample_data) - snippet_length + 1, snippet_length):
            if np.isnan(np.sum(sample_data[i:i+snippet_length])):
                continue
            snippet_data.append(sample_data[i:i+snippet_length])
            # snippet_labels.append(sample_label[i:i+snippet_length]) # For return_sequence = true
            snippet_labels.append(sample_label[i]) # For return_sequence = false

    return np.array(snippet_data), np.array(snippet_labels)

def train_test_split(data: list, labels: list, test_size=0.2, random_seed=None):
    '''
    Description:  Split train test data and labels. Similar implementations to sklearn train_test_split. 
    Inputs:       numpy array (kinematics data)
                  numpy array (true labels)
                  int (proportion of train/test)
                  int (random seed)
    Outputs:      numpy array (   1st dimension: snippet_num, 
                                  2nd dimension: time_steps,
                                  3rd dimension: training kinematics data)
                  numpy array (   1st dimension: snippet_num, 
                                  2nd dimension: time_steps,
                                  3rd dimension: testing kinematics data)
                  numpy array (   1st dimension: snippet_num, 
                                  2nd dimension: time_steps,
                                  3rd dimension: training labels)
                  numpy array (   1st dimension: snippet_num, 
                                  2nd dimension: time_steps,
                                  3rd dimension: testing labels)
    '''
    if random_seed is not None:
        np.random.seed(random_seed)
        
    indices = np.random.permutation(len(data))
    shuffled_data = data[indices]
    shuffled_labels = labels[indices]
    
    split_idx = int(len(data) * (1 - test_size))
    train_data, test_data = shuffled_data[:split_idx], shuffled_data[split_idx:]
    train_labels, test_labels = shuffled_labels[:split_idx], shuffled_labels[split_idx:]

    return train_data, test_data, train_labels, test_labels

def normalize_data(data):
    """
    Normalize the data excluding NaN values.

    Parameters:
    - data: NumPy array of shape (samples, timesteps, features).

    Returns:
    - Normalized data with the same shape, where normalization is applied only to non-NaN values.
    """
    # Initialize a copy of the data to avoid modifying the original array
    normalized_data = np.copy(data)

    # Calculate mean and std only for non-NaN values
    mean = np.nanmean(normalized_data, axis=(0, 1), keepdims=True)
    std = np.nanstd(normalized_data, axis=(0, 1), keepdims=True)
    
    # Avoid division by zero in case of constant features
    std[std == 0] = 1
    
    # Apply normalization only where data is not NaN
    np.putmask(normalized_data, ~np.isnan(normalized_data), (normalized_data - mean) / std)

    return normalized_data

def downsample_snippets(data, original_rate=30, target_rate=6):
    """
    Downsamples the data from an original rate to a target rate.

    Parameters:
    - data: Input data of shape (snippet_num, time_steps, features).
    - original_rate: The original data collection rate (Hz).
    - target_rate: The desired data collection rate (Hz).

    Returns:
    - Downsampled data of shape (snippet_num, new_time_steps, features).
    """
    factor = original_rate // target_rate
    # Take every 'factor'-th timestep in the sequence
    downsampled_data = data[:, ::factor, :]
    return downsampled_data

def count_gestures_from_snippets(snippet_labels, snippet_length):
    """
    Counts the number of gestures in snippets based on one-hot encoded labels.

    Parameters:
    - snippet_labels: A NumPy array with dimensions (snippet_num, gesture_types),
      where each row is a one-hot encoded vector representing the gesture label of a snippet.

    Returns:
    - A NumPy array with counts for each gesture type.
    """
    # Sum the one-hot encoded vectors across the snippets to get gesture counts
    gesture_counts = np.sum(snippet_labels, axis=0) * snippet_length
    
    return gesture_counts

def display_gesture_counts(gesture_counts_array, gesture_mapping):
    """
    Display gesture counts with their actual names.

    Parameters:
    - gesture_counts_array: A NumPy array with counts for each gesture type.
    - gesture_index_to_name: A dictionary mapping gesture indices to their actual names.
    """
    for index, count in enumerate(gesture_counts_array):
        gesture_name = gesture_mapping.get(index, "Unknown Gesture")
        print(f"{gesture_name}: {int(count)}")

def preprocess():
    '''
    Description:  Preprocess function, to be used in main.py. 
    Inputs:       None
    Outputs:      numpy array (   1st dimension: snippet_num, 
                                  2nd dimension: time_steps,
                                  3rd dimension: training kinematics data)
                  numpy array (   1st dimension: snippet_num, 
                                  2nd dimension: time_steps,
                                  3rd dimension: testing kinematics data)
                  numpy array (   1st dimension: snippet_num, 
                                  2nd dimension: time_steps,
                                  3rd dimension: training labels)
                  numpy array (   1st dimension: snippet_num, 
                                  2nd dimension: time_steps,
                                  3rd dimension: testing labels)  
    '''
    # Base paths (CHANGE TO BASE FOLDER PATH IF USING COMMENT LINE)
    dirname = os.path.dirname(__file__)
    # filename = os.path.join(dirname, 'Suturing')
    kinematics_base_path = dirname
    transcription_base_path = dirname

    combined_kinematics = {}
    combined_transcription = {}

    for activity in ACTIVITIES:
        # All data
        # kinematics_folder_path = os.path.join(kinematics_base_path, activity, 'kinematics', 'AllGestures')
        # transcription_folder_path = os.path.join(transcription_base_path, activity, 'transcriptions')

        # Sample data
        kinematics_folder_path = os.path.join(kinematics_base_path, activity, 'kinematics', 'AllGestures')
        transcription_folder_path = os.path.join(transcription_base_path, activity, 'transcriptions')
        
        kinematics = read_kinematic_files_from_folder(kinematics_folder_path)
        transcription = read_transcription_files_from_folder(transcription_folder_path)
        
        combined_kinematics.update(kinematics)
        combined_transcription.update(transcription)

    combined_kinematics = del_not_transcripted_kinematics(combined_kinematics, combined_transcription)
    combined_kinematics = trim_kinematics_arr(combined_kinematics, combined_transcription)
    true_label, gesture_counts = generate_labels(combined_transcription)
    full_data_arr = stack_dictionary_val(combined_kinematics)
    
    full_label_arr = stack_dictionary_val(true_label)
    full_data_arr, full_label_arr = split_into_snippets(full_data_arr, full_label_arr, SNIPPET_LEN)
    full_label_arr = full_label_arr.astype(int)
    train_data, test_data, train_labels, test_labels = train_test_split(full_data_arr, full_label_arr, 0.2, 41)
    return train_data, test_data, train_labels, test_labels, gesture_counts

def preprocessing_one_trial_out():
    code_path = os.path.dirname(__file__)
    trial_folder = os.path.join(code_path, "Experimental_setup", "Suturing", "Balanced", "GestureClassification", "UserOut", "1_Out", "itr_1")

    # Read the trial data
    train_data, test_data = read_one_trial_out_folder(trial_folder)
    # displayData("train", 0)

    train_kinematic_data = extract_kinematic_data(code_path, train_data)
    test_kinematic_data = extract_kinematic_data(code_path, test_data)
    # gesture_counts = count_gestures(train_kinematic_data)
    # print("test")
    # print(type(test_kinematic_data[0]))

    train_data_array, train_gesture_array = stack_segments(train_kinematic_data)
    test, test_labels = stack_segments(test_kinematic_data)
    # print(test_data_array.shape)
    # print("test")
    # print(test[0][-1])
    # print("test_labels")
    # print(test_labels[0][-1])

    train_data_array = normalize_data(train_data_array)
    test = normalize_data(test)
    # print("test")
    # print(test[0][-1])

    snippet_data, snippet_label = split_into_snippets(train_data_array, train_gesture_array, SNIPPET_LEN)
    test, test_labels = split_into_snippets(test, test_labels, SNIPPET_LEN)
    # print("test")
    # print(test[1000:1001])
    # print("test_labels")
    # print(test_labels[1000:1001])
    # print(snippet_data.shape)
    # print(snippet_label.shape)
    # print(test.shape)
    # print(test_labels.shape)
    snippet_data = downsample_snippets(snippet_data, original_rate=SNIPPET_LEN, target_rate=DOWNSAMPLE_LEN)
    test = downsample_snippets(test, original_rate=SNIPPET_LEN, target_rate=DOWNSAMPLE_LEN)
    train_gesture_counts = count_gestures_from_snippets(snippet_label, DOWNSAMPLE_LEN)
    test_gesture_counts = count_gestures_from_snippets(test_labels, DOWNSAMPLE_LEN)
    # Downsample labels if return_sequence = true
    # snippet_label = downsample_snippets(snippet_label, original_rate=30, target_rate=6)
    # test_labels = downsample_snippets(test_labels, original_rate=30, target_rate=6)
    # print("test")
    # print(test[1000:1001])
    # print("test_labels")
    # print(test_labels[1000:1001])
    train, validation, train_labels, validation_labels = train_test_split(snippet_data, snippet_label, test_size=0.2, random_seed=41)
    # print(train.shape)
    # print(validation.shape)
    # print(train_labels.shape)
    # print(validation_labels.shape)

    return train, validation, test, train_labels, validation_labels, test_labels, train_gesture_counts, test_gesture_counts
