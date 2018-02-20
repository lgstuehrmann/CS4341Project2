from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
#Img Preprocessing
batch_size = 512
epochs=10


#Begin Model
model = Sequential()
model.add(Dense(10, input_shape=(28*28, ), kernel_initializer='he_normal'))
model.add(Activation('relu'))

model.add(Dense(10, kernel_initializer='he_normal'))
model.add(Activation('softmax'))

# Compile Model
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

#Fit model
history=model.fit(x_train, y_train, validation_data = (x_val, y_val),
                  epochs=epochs, batch_size=batch_size)
#Get Score
score=model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy: ', score[1])
print(history.history)
