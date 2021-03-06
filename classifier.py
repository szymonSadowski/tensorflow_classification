import tensorflow as tf
import numpy as np
from tensorflow import keras

# * as for 03.03 it needes to be matplotlib 3.2
import matplotlib.pyplot as plt

# * Load dataset provided by keras https://keras.io/api/datasets/fashion_mnist/
fashion_mnist = keras.datasets.fashion_mnist

# * Get data
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


# plt.imshow(train_images[0], cmap="gray", vmin=0, vmax=255)
# plt.show()

# * model
brain = keras.Sequential(
    [
        # * flatten the data 1 layer (input layer)
        keras.layers.Flatten(input_shape=(28, 28)),
        # * 2 layer gives propability of what item it is returns 0 to 1
        keras.layers.Dense(128, activation=tf.nn.relu),
        # * 3 layer outpus propabiolity of item 0-9 (because we have that many clothing items in dataset)
        keras.layers.Dense(10, activation=tf.nn.softmax),
    ]
)
brain.compile(
    optimizer=tf.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)
brain.fit(train_images, train_labels, epochs=6)


test = brain.evaluate(test_images, test_labels)

plt.imshow(test_images[100], cmap="gray", vmin=0, vmax=255)
plt.show
pred = brain.predict(test_images)
print(list(pred[100]).index(max(pred[100])))
print(test_labels[100])
msg = "Code Completed"
print(msg)