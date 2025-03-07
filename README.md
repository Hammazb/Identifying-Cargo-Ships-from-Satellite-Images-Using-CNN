# Identifying-Cargo-Ships-from-Satellite-Images-Using-CNN

This project focuses on building a deep learning model to classify ship images using Convolutional Neural Networks (CNNs). The dataset, stored in a JSON file, contains images represented as numerical arrays with corresponding labels, where 0 represents "No Ship" and 1 represents "Ship." The goal is to develop a robust model that accurately detects ships in images, which can be applied to maritime surveillance, environmental monitoring, and security applications.

The dataset is preprocessed by converting it into NumPy arrays, normalizing pixel values, and reshaping images to a uniform size of 80x80 pixels. To ensure fair training, the dataset is split into training (80%) and testing (20%) sets, with class balancing applied to prevent bias toward more frequent categories. Data augmentation techniques such as rotation, shifting, and flipping are used to improve generalization.

The CNN model is built using the Keras Sequential API, consisting of multiple convolutional layers that extract key image features, followed by pooling layers to reduce dimensionality. Dropout layers are incorporated to prevent overfitting, and fully connected dense layers process the extracted features to classify images. The final softmax activation layer outputs probabilities for two categories: "Ship" and "No Ship." The model is compiled using the Adam optimizer with a learning rate of 0.001 and trained with categorical cross-entropy as the loss function.

To enhance model performance, callbacks such as ReduceLROnPlateau and ModelCheckpoint are used. ReduceLROnPlateau dynamically adjusts the learning rate when validation accuracy stops improving, ensuring efficient learning, while ModelCheckpoint saves the best-performing model during training. The model is trained over 20 epochs, and its performance is evaluated using accuracy and loss curves, confusion matrices, and classification reports.

The results demonstrate that the CNN effectively classifies ships with high accuracy, making it suitable for real-world applications. Potential business applications include maritime security, where unauthorized vessels can be detected, environmental monitoring to track ships responsible for ocean pollution, and autonomous navigation systems for AI-powered vessels. Further improvements could be achieved by incorporating transfer learning with pre-trained models such as ResNet or MobileNet, fine-tuning hyperparameters, and expanding the dataset for greater diversity. This project highlights the potential of deep learning in image classification tasks and its impact on various industries.

Reference from: 
https://www.kaggle.com/code/paultimothymooney/identifying-ships-in-satellite-images
