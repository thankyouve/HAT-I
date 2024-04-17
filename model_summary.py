from tensorflow.keras.models import load_model

# Load the model from the .h5 file
model = load_model('E:/School/McmasterU/HAT&I/Model/JIGSAWS_LSTM/JIGSAWS_LSTM_ec.h5')

# Print the model architecture
model.summary()

for layer in model.layers:
    # Get the configuration of the specific layer
    config = layer.get_config()
    print(config)
