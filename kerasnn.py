import keras as k
import numpy as np
import sklearn as scikit

model = k.Sequential()
model.add(k.Dense(10, input_shape=(28*28, ), kernel_initializer='he_normal'))
model.add(k.Activation('relu'))







model.add(k.Dense(10, kernel_initializer='he_normal'))
model.add(k.Activation('softmax'))

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

history=model.fit(x_train, y_train, validation_data = (x_val, y_val),
                  epochs = 10, batch_size=512)

print(history.history)
