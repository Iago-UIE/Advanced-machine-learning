import numpy as np
import keras
from keras import layers
import pandas as pd

# It has 10 classes:
'''
0 T-shirt/top
1 Trouser
2 Pullover
3 Dress
4 Coat
5 Sandal
6 Shirt
7 Sneaker
8 Bag
9 Ankle boot
'''
# Neurons per class
num_classes = 10
# Each image has 28x28 pixels and, being grayscale, has 1 channel
input_shape = (28, 28, 1)

# Load data
train_df = pd.read_csv("fashion-mnist_train.csv")
test_df = pd.read_csv("fashion-mnist_test.csv")

# Extract features and labels
X_Train = train_df.drop(columns="label").values
Y_Train = train_df["label"].values
X_Test = test_df.drop(columns="label").values
Y_Test = test_df["label"].values

# Normalize and reshape the data
X_Train = X_Train.astype("float32") / 255.0
X_Test = X_Test.astype("float32") / 255.0

# -1: automatically infer the number of samples
# 28, 28: width and height of the images
# 1: grayscale
X_Train = X_Train.reshape(-1, 28, 28, 1)  # Reshape to (60000, 28, 28, 1)
X_Test = X_Test.reshape(-1, 28, 28, 1)     # Reshape to (10000, 28, 28, 1)

# Convert labels to categorical matrices (one-hot vectors with 0s and 1s)
y_train = keras.utils.to_categorical(Y_Train, num_classes)
y_test = keras.utils.to_categorical(Y_Test, num_classes)

# 32: Number of filters (feature detectors)
# Kernel size: size of the convolution window
# Pool size: reduces the spatial dimensions
# Flatten() flattens the data to connect to dense layers
# Dropout(0.5) prevents overfitting
# Build the model
model = keras.Sequential([
    keras.Input(shape=input_shape),
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation="softmax")
])

model.summary()

# Train the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X_Train, y_train, batch_size=128, epochs=15, validation_split=0.1)

# Evaluate the model
score = model.evaluate(X_Test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])


### Test loss: 0.24317917227745056
### Test accuracy: 0.9132999777793884

### Since we have a test loss lower than 0.3, we assume that we do not need to change the value of layers.Dropout as we have a good model
