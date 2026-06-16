import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Activation, Dropout, Add, GlobalAveragePooling1D, Dense, \
    BatchNormalization
import os


# Define the TCN model architecture as a function
def build_tcn_model(input_shape):
    def residual_block(x, filters, kernel_size, dilation_rate):
        shortcut = x

        # First convolutional layer
        conv1 = Conv1D(filters, kernel_size, dilation_rate=dilation_rate, padding='causal')(x)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        conv1 = Dropout(0.2)(conv1)

        # Second convolutional layer
        conv2 = Conv1D(filters, kernel_size, dilation_rate=dilation_rate, padding='causal')(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)
        conv2 = Dropout(0.2)(conv2)

        # Ensure the shortcut has a matching number of filters
        if shortcut.shape[-1] != filters:
            shortcut = Conv1D(filters, 1, padding='same')(shortcut)

        res_output = Add()([shortcut, conv2])
        return res_output

    input_layer = Input(shape=input_shape)

    # Build the TCN model with a series of residual blocks
    x = Conv1D(32, kernel_size=3, padding='causal')(input_layer)
    x = residual_block(x, filters=32, kernel_size=3, dilation_rate=1)
    x = residual_block(x, filters=64, kernel_size=3, dilation_rate=2)
    x = residual_block(x, filters=128, kernel_size=3, dilation_rate=4)
    x = residual_block(x, filters=256, kernel_size=3, dilation_rate=8)

    # Final layers for classification
    x = GlobalAveragePooling1D()(x)
    output_layer = Dense(1, activation='sigmoid')(x)

    return Model(inputs=input_layer, outputs=output_layer)


# 1. Load your pre-processed data from Block 1
# Replace 'path/to/your/files/' with the actual path to your .npy files
try:
    segments = np.load('segments.npy')
    labels = np.load('labels.npy')
except FileNotFoundError:
    print("Error: The segments.npy or labels.npy file was not found.")
    print("Please ensure your pre-processed data files are in the correct location.")
    exit()

# 2. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(segments, labels, test_size=0.2, random_state=42)

# 3. Reshape and convert data to TCN input format
# The TCN expects a shape of (num_samples, timesteps, features)
X_train = np.expand_dims(X_train, axis=-1).astype(np.float32)
X_test = np.expand_dims(X_test, axis=-1).astype(np.float32)

y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)

# 4. Define and Compile the TCN model
input_shape = (X_train.shape[1], X_train.shape[2])
tcn_model = build_tcn_model(input_shape)

tcn_model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=['accuracy'])
print("Trainable Parameters:", tcn_model.count_params())
tcn_model.summary()

# 5. Train the model
batch_size = 32
epochs = 15
print("\nTraining the TCN model...")
tcn_model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))

# 6. Evaluate the float model
score = tcn_model.evaluate(X_test, y_test, verbose=0)
print("\n----------------------------------------------------")
print("Evaluation of the original (float32) model:")
print(f"Test Loss: {score[0]:.4f}")
print(f"Test Accuracy: {score[1] * 100:.2f}%")
print("----------------------------------------------------")

# 7. Quantize the model
# Prepare the directory for saving the quantized model
export_path = 'model/tf/'
os.makedirs(export_path, exist_ok=True)

# Use the original TensorFlow Lite converter logic
print("\nConverting model to a quantized (int8) version...")
converter = tf.lite.TFLiteConverter.from_keras_model(tcn_model)
converter.experimental_new_converter = True


def representative_dataset_gen():
    num_calibration_steps = 100
    for i in range(num_calibration_steps):
        yield ([np.expand_dims(X_train[i], axis=0)])


converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
converter.representative_dataset = representative_dataset_gen
tflite_quantized_model = converter.convert()

# Save the quantized model
tflite_model_name = 'ECG-TCN_quantized'
open(export_path + tflite_model_name + '.tflite', 'wb').write(tflite_quantized_model)
print(f"Quantized model saved to: {export_path}{tflite_model_name}.tflite")

# 8. Evaluate the quantized model
print("\nEvaluating the quantized model...")
interpreter = tf.lite.Interpreter(model_content=tflite_quantized_model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

predictions_quantized = np.zeros(len(X_test), dtype=int)
input_scale, input_zero_point = input_details[0]["quantization"]

for i in range(len(X_test)):
    input_data = X_test[i]
    input_data = input_data / input_scale + input_zero_point
    input_data = np.expand_dims(input_data, axis=0).astype(input_details[0]["dtype"])

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    predictions_quantized[i] = np.argmax(output)

accuracy_quantized = np.mean(predictions_quantized == y_test)
print("----------------------------------------------------")
print(f"Evaluation of the quantized (int8) model:")
print(f"Test Accuracy: {accuracy_quantized * 100:.2f}%")
print("----------------------------------------------------")
print("\nQuantization results:")
print("Original float32 accuracy: {:.2f}%".format(score[1] * 100))
print("Quantized int8 accuracy: {:.2f}%".format(accuracy_quantized * 100))
print("Accuracy change: {:.2f}%".format((accuracy_quantized - score[1]) * 100))