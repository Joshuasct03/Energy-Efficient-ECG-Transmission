# Energy-Efficient Adaptive ECG Transmission for Wearable Healthcare Monitoring

## Overview

This project presents an intelligent ECG monitoring framework designed for energy-efficient wearable healthcare systems. The system combines deep learning-based ECG classification with adaptive signal processing and communication techniques to reduce transmission energy while maintaining signal reconstruction quality.

The framework uses a Temporal Convolutional Network (TCN) model to analyze ECG segments and perform adaptive decision-making for optimized data transmission in resource-constrained environments.

## Features

- ECG signal preprocessing and segmentation
- ECG classification using Temporal Convolutional Network (TCN)
- Adaptive ECG compression based on signal characteristics
- TensorFlow Lite optimization for edge deployment
- Adaptive transmission strategy for energy-efficient communication
- Performance evaluation using classification and reconstruction metrics

## System Workflow

1. ECG signal acquisition and preprocessing
2. ECG segment classification using the TCN model
3. Adaptive parameter selection based on ECG characteristics and system conditions
4. ECG compression and transmission simulation
5. Signal reconstruction and performance evaluation

## Machine Learning Model

The project implements a deep learning-based ECG classifier.

Model Details:
- Architecture: Temporal Convolutional Network (TCN)
- Framework: TensorFlow and TensorFlow Lite
- Input: ECG signal segments
- Task: Normal and arrhythmic ECG classification

Evaluation Metrics:
- Accuracy
- Precision
- Recall
- F1-score

## Dataset

The system was evaluated using the MIT-BIH Arrhythmia Database, a standard benchmark dataset used for ECG analysis research.

ECG signals were preprocessed and converted into segments suitable for machine learning-based classification and transmission experiments.

## Technologies Used

- Python
- TensorFlow
- TensorFlow Lite
- NumPy
- SciPy
- Scikit-learn
- Signal Processing Algorithms
- Machine Learning
- Deep Learning
- Edge AI Concepts

## Performance Evaluation

The system performance was analyzed using:

- ECG classification accuracy
- Signal reconstruction quality
- Transmission energy consumption
- Adaptive system behavior under varying conditions

The adaptive approach reduces transmission energy requirements while maintaining ECG reconstruction quality.

## Applications

- Wearable healthcare monitoring systems
- Remote health monitoring
- Edge AI applications
- IoT-based healthcare devices

## Team

Developed as an academic research project by a four-member team from the Department of Electronics and Communication Engineering, Sree Chitra Thirunal College of Engineering.

## Status

Accepted for presentation at the 2026 International Conference on Smart Communication and Sustainable Technologies (ICSCST).
