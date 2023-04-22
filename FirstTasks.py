import tensorflow as tf
import numpy as np
import mnist

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from matplotlib import pyplot
from keras.utils import to_categorical

from tensorflow import keras
from keras import backend as K

train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5

train_images = train_images.reshape((-1, 784))
test_images = test_images.reshape((-1, 784))

model = Sequential([
  Dense(64, activation='relu', input_shape=(784,)),
  Dense(64, activation='relu'),
  Dense(10, activation='softmax'),
])

model.compile(
  optimizer='adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

# This will stop the model if it stops improving to avoid overfitting
f1 = EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=10)

# Used to save the model at certain intervals during training
f2 = ModelCheckpoint(filepath="model.h5", monitor='val_loss', mode='min', verbose=2, save_best_only=True)

'''
For model.fit() "verbose" defines how information will be printed to the console during training:
- 0 means that nothing will be printed to the console during training
- 1 means that a progress bar will be displayed, which shows the current epoch number and the progress within the epoch (i.e., percentage of completion)
- 2 means that a summary line will be printed after each epoch, showing the training loss and any other metrics that have been specified

BUT for EarlyStopping and ModelCheckpoint it defines how much information will be printed:
0 - silent
1 - message printed when an event occurs
2 - a more detailed message printed
'''

history = model.fit(
  train_images,
  to_categorical(train_labels),
  epochs=100,
  batch_size=30,
  validation_data=(test_images, to_categorical(test_labels)),
  verbose=1,
  callbacks=[f1, f2]
).history

'''
TASK 7

When I ran the model it stopped after 24 epochs because val_loss did not improve for 10 consecutive epochs(because EarlyStopping "patience"
is set to 10)
'''


print(model.summary())

# NOTE: same name as in ModelCheckpoint
trained_model = load_model("model.h5")

train_loss, train_accuracy = trained_model.evaluate(train_images, to_categorical(train_labels))
# TASK 8
print("Training loss:", train_loss)
print("Training accuracy:", train_accuracy)

test_loss, test_accuracy = trained_model.evaluate(test_images, to_categorical(test_labels))
# TASK 8
print("Testing loss:", train_loss)
print("Testing accuracy:", train_accuracy)

# Plot the training and validation accuracy
pyplot.plot(history['accuracy'])
pyplot.plot(history['val_accuracy'])
pyplot.title('Model accuracy')
pyplot.ylabel('Accuracy')
pyplot.xlabel('Epoch')
pyplot.legend(['Train', 'Validation'], loc='upper left')
# saving the image
pyplot.savefig('accuracy_plot.png')
pyplot.show()

# Plot the training and validation loss
pyplot.plot(history['loss'])
pyplot.plot(history['val_loss'])
pyplot.title('Model loss')
pyplot.ylabel('Loss')
pyplot.xlabel('Epoch')
pyplot.legend(['Train', 'Validation'], loc='upper left')
# saving the image
pyplot.savefig('loss_plot.png')
pyplot.show()
