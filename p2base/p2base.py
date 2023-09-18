import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

INPUTS = np.array([(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0), (0, 0, 1, 1), (0, 1, 0, 0), (0, 1, 0, 1), (0, 1, 1, 0), (0, 1, 1, 1),
                   (1, 0, 0, 0), (1, 0, 0, 1), (1, 0, 1, 0), (1, 0, 1, 1), (1, 1, 0, 0), (1, 1, 0, 1), (1, 1, 1, 0), (1, 1, 1, 1)])

Y = np.array([[0], [1], [0], [0], [1], [1], [1], [0],
             [0], [0], [0], [0], [1], [1], [0], [0]])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def forward(tupla):

    tupla = np.asarray(tupla)

    w1 = np.array([[-80, 20, 20, -80],
                   [-80, 20, -80, -80],
                   [-80, -80, -80, 20],
                   [-80, 40, -80, 20],
                   [20, 20, -80, 20],
                   [30, 30, -80, 10]])

    Bias = np.array([-30, -10, -10, -30, -50, -60])
    BiasOculta = np.array([-10])
    w2 = np.array([30, 30, 30, 30, 30, 30])
    CapaOculta = []

    for i in range(6):
        auxiliar = w1[i]
        x = sum(tupla * auxiliar)
        resultado = sigmoid(x + Bias[i])
        CapaOculta.append(resultado)

    CapaOculta = np.asarray(CapaOculta)
    output = sigmoid(sum(CapaOculta * w2) + BiasOculta[0])
    output = output

    return format(output)


def keras(X, Y, EPOCHS, BATCH):

    model = Sequential()
    model.add(Dense(6, input_shape=(4,), activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))

    adam = Adam(learning_rate=0.1)
    model.compile(loss='mean_squared_error', optimizer=adam, metrics=['acc'])
    model.fit(X, Y, epochs=EPOCHS, batch_size=BATCH)

    z = model.predict(X)
    print()
    for i, j in zip(Y, z):
        print('{} => {}'.format(i, j))
    print()

    Keras_w1 = model.layers[0].get_weights()[0]
    Keras_b1 = model.layers[0].get_weights()[1]
    Keras_w2 = model.layers[1].get_weights()[0]
    Keras_b2 = model.layers[1].get_weights()[1]

    print("Pesos capa 1: ")
    print(Keras_w1, "\n")
    print("Bias capa 1: ")
    print(Keras_b1, "\n")
    print("Pesos capa 2: ")
    print(Keras_w2, "\n")
    print("Bias capa 2: ")
    print(Keras_b2, "\n")


if __name__ == "__main__":
    print()
    input("Pulse intro para probar el MLP...")
    print()

    for i in range(16):
        print(str(INPUTS[i]) + " -> " + forward((INPUTS[i])))

    print()
    input("Pulse intro para probar Keras...")
    print()

    keras(INPUTS, Y, 150, 16)
