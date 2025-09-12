import numpy as np
import keras
from keras import layers, ops
import pickle

def nn_dense_layers(x_train, y_train, x_test, y_test):

    # Flatten the data and normalize it to a smaller scale
    x_train = x_train.reshape(60000, 784).astype("float32") / 255
    x_test = x_test.reshape(10000, 784).astype("float32") / 255


    inputs = keras.Input(shape=(784,))
    dense = layers.Dense(128, activation='relu')
    x = dense(inputs)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(10)(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
    model.summary()

    # Fit the data into our neural model
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer = keras.optimizers.RMSprop(),
        metrics  =["accuracy"],
    )

    history = model.fit(x_train, y_train, batch_size=128, epochs=15, validation_split=0.2)
    test_scores = model.evaluate(x_test, y_test, verbose=2)
    predictions = model.predict(x_test)

    # Get the index of max prediction
    pred_class = np.argmax(predictions, axis=1)

    print("Test loss",test_scores[0])
    print("Test accuracy", test_scores[1])

    np.savetxt('PRED_mlp.dat', pred_class, fmt='%d')

def CNN(x_train, y_train, x_test, y_test):

    inputs = keras.Input(shape=(28, 28, 1))

    x = layers.Conv2D(32, (3, 3), activation="relu")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(10)(x)

    model = keras.Model(inputs, outputs, name="fashion_cnn")

    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer = keras.optimizers.RMSprop(),
        metrics  =["accuracy"],
    )

    history = model.fit(x_train, y_train, batch_size=128, epochs=10, validation_split=0.2)
    test_scores = model.evaluate(x_test, y_test, verbose=2)

    print("Test loss",test_scores[0])
    print("Test accuracy", test_scores[1])


def main():

    # Load Fashion-MNIST data
    with open('mnist_fashion.pkl', 'rb') as f:
        x_train = pickle.load(f)
        y_train = pickle.load(f)
        x_test = pickle.load(f)
        y_test = pickle.load(f)

    CNN(x_train, y_train, x_test, y_test)

if __name__ == '__main__':
    main()
