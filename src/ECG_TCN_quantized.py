import tensorflow as tf
import numpy as np
import os

class ECGClassifier:
    def __init__(self, model_path='models/ECG-TCN_quantized.tflite'):
        """Load the quantized TFLite model for inference."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}. Run pipeline in training mode first.")
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print(f"Loaded TFLite model from {model_path} (ready for inference)")

    def predict(self, X):
        """Predict on ECG segments. X: (N, 512, 1) float32 → probabilities (N,)."""
        if len(X.shape) != 3 or X.shape[1:] != (512, 1):
            raise ValueError("Input must be (N, 512, 1) float32 array.")
        predictions = []
        for i in range(len(X)):
            # Prepare input (batch of 1)
            input_tensor = np.expand_dims(X[i], axis=0).astype(np.float32)
            self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
            # Run inference
            self.interpreter.invoke()
            # Get sigmoid probability
            output = self.interpreter.get_tensor(self.output_details[0]['index'])
            pred_prob = float(output[0][0])  # Scalar from (1,1)
            predictions.append(pred_prob)
        return np.array(predictions)  # Returns probs (N,)
