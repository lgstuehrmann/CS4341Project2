"""
Robert Harrison
Lucy Stuehrmann
Brady Snowden
"""

from keras.models import Sequential
from keras.layers import Dense, Activation
from tools import preprocess, confusion
import matplotlib.pyplot as plot

# Img Preprocessing
batch_size = 512
epochs = 2000

x_train, x_test, x_val, y_train, y_test, y_val = preprocess()

# Begin Model
model = Sequential()
model.add(Dense(20, input_shape=(28 * 28,), kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(Dense(15, kernel_initializer='he_normal'))
model.add(Activation('selu'))
model.add(Dense(12, kernel_initializer='glorot_uniform'))
model.add(Activation('tanh'))
model.add(Dense(10, kernel_initializer='he_normal'))
model.add(Activation('softmax'))

# Compile Model
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit model
history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                    epochs=epochs, batch_size=batch_size)
# Get Score
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy: ', score[1])
print(history.history)

plot.plot(history.history['acc'])
plot.plot(history.history['val_acc'])
plot.title('Model Accuracy')
plot.ylabel('Accuracy')
plot.xlabel('Epoch')
plot.legend(['Train', 'Test'], loc='lower right')
plot.show()

plot.plot(history.history['loss'])
plot.plot(history.history['val_loss'])
plot.title('Model Loss')
plot.ylabel('Loss')
plot.xlabel('Epoch')
plot.legend(['Train', 'Test'], loc='upper right')
plot.show()

predictions = model.predict(x_test, verbose=0)
matrix = confusion(y_test, predictions)

figure, ax = plot.subplots()
plot.ylabel('Predictions')
plot.xlabel('Actual')
plot.title('Confusion Matrix for Keras Neural Net')
plot.xticks([0,1,2,3,4,5,6,7,8,9])
plot.yticks([0,1,2,3,4,5,6,7,8,9])

ax.matshow(matrix, cmap=plot.cm.Spectral)
x = [0,1,2,3,4,5,6,7,8,9]
for i in x:
    for j in x:
        c = matrix[j,i]
        ax.text(i,j,str(c), va='center', ha='center')

plot.show()
print(matrix)

