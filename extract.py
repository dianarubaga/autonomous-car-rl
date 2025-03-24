from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

# Define a dictionary for custom objects
custom_objects = {"mse": MeanSquaredError()}  # Fix for missing loss function

# Load the model using custom_objects
model_path = "dqn_models/dqn_model.h5"

try:
    model = load_model(model_path, custom_objects=custom_objects)
    print("✅ Model loaded successfully!\n")

    # Loop through each layer and print weights
    for layer in model.layers:
        weights = layer.get_weights()
        print(f"🔹 Layer: {layer.name}")
        if weights:  # If the layer has trainable parameters
            print(f"  - Weights shape: {weights[0].shape}")
            print(f"  - Bias shape: {weights[1].shape}")
            print(f"  - Sample weights: {weights[0].flatten()[:5]}")  # Print first 5 weights
            print(f"  - Sample biases: {weights[1].flatten()[:5]}")   # Print first 5 biases
        else:
            print("  - No trainable weights in this layer.")
        print("\n" + "-"*50 + "\n")  # Separator

except Exception as e:
    print(f"🚨 Error loading model: {e}")
