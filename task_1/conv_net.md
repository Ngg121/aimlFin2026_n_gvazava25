A Convolutional Neural Network (CNN) is a deep learning architecture designed to process structured grid-like data such as images. In the illustrated architecture, the network begins with an input image, which is analyzed using small filters called kernels. These kernels slide across the image and perform convolution operations, extracting local patterns such as edges, textures, and shapes. After each convolution, a ReLU activation function is applied to introduce non-linearity, allowing the model to learn complex representations.

<img width="1400" height="658" alt="image" src="https://github.com/user-attachments/assets/cbc896ee-e022-425a-8303-cb75756022af" />


Following convolution, pooling layers reduce the spatial dimensions of the feature maps. This step simplifies the data, lowers computational cost, and helps the network focus on the most important features while reducing overfitting. As the data moves deeper into the network, multiple convolution and pooling layers progressively capture more abstract and higher-level features.

Once feature extraction is complete, the resulting feature maps are flattened into a one-dimensional vector. This vector is passed into fully connected layers, where classification takes place. Finally, a SoftMax activation function produces a probabilistic output, assigning likelihoods to each possible class. The class with the highest probability is selected as the final prediction.

<img width="1240" height="1000" alt="image" src="https://github.com/user-attachments/assets/4d23d0cf-f6f3-4e9f-b7ef-65aafe18cab8" />

The second visual illustrates a traditional fully connected neural network, also known as a multilayer perceptron (MLP). In this architecture, every neuron in one layer is connected to every neuron in the next layer, forming dense connections throughout the network. The model begins with an input layer that receives numerical features, which are then passed through multiple hidden layers. Each hidden layer transforms the input using weighted sums followed by activation functions, allowing the network to learn increasingly abstract representations of the data. As information flows forward, the network adjusts its internal weights during training using backpropagation to minimize prediction error.

The presence of three hidden layers suggests a deep architecture capable of modeling complex, non-linear relationships. Each neuron aggregates signals from all neurons in the previous layer, making the network powerful but computationally intensive compared to convolutional networks. Finally, the output layer produces the final predictions, often corresponding to class probabilities or regression values. This type of fully connected network is widely used for structured data classification, risk analysis, anomaly detection, and various cybersecurity applications such as intrusion detection and malware classification when features are pre-engineered rather than spatially structured.




Practical CNN example:

duration,packets,bytes,syn,ack,failed_logins,label
2,40,5000,0,1,0,0
1,30,4200,0,1,0,0
3,60,7000,0,1,0,0
2,55,6500,0,1,1,0
1,25,3900,0,1,0,0
4,80,9000,0,1,0,0
10,400,60000,1,0,12,1
8,350,52000,1,0,10,1
12,500,75000,1,0,15,1
9,420,68000,1,0,9,1
11,460,71000,1,0,14,1
7,300,48000,1,0,8,1



import numpy as np
import pandas as pd
from io import StringIO

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras import layers, models


1) Tiny dataset (inline)

csv_data = """duration,packets,bytes,syn,ack,failed_logins,label
2,40,5000,0,1,0,0
1,30,4200,0,1,0,0
3,60,7000,0,1,0,0
2,55,6500,0,1,1,0
1,25,3900,0,1,0,0
4,80,9000,0,1,0,0
10,400,60000,1,0,12,1
8,350,52000,1,0,10,1
12,500,75000,1,0,15,1
9,420,68000,1,0,9,1
11,460,71000,1,0,14,1
7,300,48000,1,0,8,1
"""

df = pd.read_csv(StringIO(csv_data))

X = df.drop("label", axis=1).values
y = df["label"].values


 2) Scale features

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


3) Reshape for CNN
 We have 6 features -> make a 2x3 "image"
 Shape: (samples, height=2, width=3, channels=1)


X_img = X_scaled.reshape(-1, 2, 3, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X_img, y, test_size=0.33, random_state=42, stratify=y
)


 4) Build a small CNN

model = models.Sequential([
    layers.Conv2D(8, (2, 2), activation="relu", input_shape=(2, 3, 1)),
    layers.Flatten(),
    layers.Dense(8, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


5) Train and evaluate

history = model.fit(X_train, y_train, epochs=30, verbose=0)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {acc:.3f}")


6) Predict on a new "flow"
 Example: suspicious (high packets/bytes, syn=1, ack=0, failed_logins high)


new_flow = np.array([[9, 410, 65000, 1, 0, 11]])
new_flow_scaled = scaler.transform(new_flow).reshape(-1, 2, 3, 1)

prob_attack = model.predict(new_flow_scaled, verbose=0)[0][0]
print(f"Predicted attack probability: {prob_attack:.3f}")
print("Predicted label:", 1 if prob_attack >= 0.5 else 0)

<img width="337" height="121" alt="image" src="https://github.com/user-attachments/assets/3289ac2f-3555-445b-8b7f-61ed59763642" />



What this demonstrates:

We take flow features (tabular data).
We reshape them into a small 2×3 grid so a CNN can run convolution.
The CNN learns to separate “benign-like” patterns from “attack-like” patterns.
Output is an attack probability (sigmoid).


