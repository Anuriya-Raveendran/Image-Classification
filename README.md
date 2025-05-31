# Image-Classification
This project focuses on image classification of traffic signs using Convolutional Neural Networks (CNNs). The goal is to accurately recognize and classify road signs from images into one of the predefined classes, helping to lay the foundation for applications in autonomous driving and intelligent transportation systems.

🧠 Problem Statement
Traffic signs play a critical role in road safety. Automatic recognition of traffic signs from images allows for enhanced decision-making in autonomous vehicles. This project aims to classify road sign images into their correct categories using a deep learning model based purely on image data.


🧱 Model Architecture (CNN)
The CNN model is structured as follows:

Input Shape: (64, 64, 3) resized RGB images

Convolutional Layers: Multiple Conv2D layers with ReLU activation

Pooling Layers: MaxPooling to reduce dimensionality

Dropout: Applied after Conv layers to prevent overfitting

Flatten Layer: Converts 2D feature maps into 1D vector

Dense Layers: Fully connected layers for final classification

Output Layer: 43 units with softmax activation for multi-class output

🏁 Model Training
Loss Function: Sparse Categorical Crossentropy

Optimizer: Adam

Batch Size: 32

Epochs: 20

Validation Strategy: 70% Train, 15% Validation, 15% Test Split

EarlyStopping and ModelCheckpoint used to improve generalization

🧪 Evaluation Metrics
Accuracy: Achieved over 94% on test data

Confusion Matrix and Classification Report used to assess per-class performance

Visualizations: Training vs Validation accuracy plots, and error analysis


🔧 Tools & Libraries
TensorFlow / Keras

NumPy, Matplotlib, Seaborn

Scikit-learn for evaluation

Jupyter Notebooks or Google Colab

🚀 Future Scope
Integrate with object detection for real-time road sign recognition

Extend to handle multilingual or region-specific traffic signs

Deploy as a mobile or web app for field use

