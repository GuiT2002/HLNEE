import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, MaxPool2D, Flatten, Conv2D, Dropout
from keras.callbacks import EarlyStopping

# Loading the dataset

(image_training, label_training), (image_test, label_test) = tf.keras.datasets.cifar100.load_data()

# Normalizing the pixels of the images

image_training = image_training/255
image_test = image_test/255

# One-hot encoding the labels of the classes

label_training = tf.keras.utils.to_categorical(label_training, 100)
label_test = tf.keras.utils.to_categorical(label_test, 100)

# Creating a Convolutional Neural Network

model = Sequential()

model.add(Conv2D(64, 3, padding='same', activation='relu'))
model.add(Conv2D(128, 3, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.4))

model.add(Conv2D(256, 3, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(768, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(100, activation='softmax'))

# Preparing the model for training

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=3)

# Using TensorBoard to visualize the training

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1)

# Training the model

trained_model = model.fit(image_training, label_training, epochs=100, batch_size=100,
                          validation_data=(image_test, label_test), callbacks=[early_stop, tensorboard_callback])

# Evaluating the model

test_loss, test_acc = model.evaluate(image_test, label_test)

print(f'The evaluated loss of the model is: {test_loss}\nThe evaluated accuracy of the model is: {test_acc}')

# Saving the model

model.save('HLN AI Model v0.2.h5')
