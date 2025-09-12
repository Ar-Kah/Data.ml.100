import numpy as np
import keras
from keras import layers, ops
import pickle

def main():
    # Load Fashion-MNIST data
    with open('mnist_fashion.pkl', 'rb') as f:
        x_train = pickle.load(f)
        y_train = pickle.load(f)
        x_test = pickle.load(f)
        y_test = pickle.load(f)

    # Flatten the data
    x_train = x_train.reshape(60000, 784).astype("float32") / 255
    x_test = x_test.reshape(10000, 784).astype("float32") / 255


    inputs = keras.Input(shape=(784,))
    dense = layers.Dense(64, activation='relu')
    x = dense(inputs)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(10)(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")

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

if __name__ == '__main__':
    main()
