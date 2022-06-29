import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input, concatenate
from tensorflow.keras import Model

import keras2onnx
import onnx
import tf2onnx.convert

batch_size = 128
num_classes = 10
epochs = 10

img_rows, img_cols = 28, 28

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.0
x_test /= 255.0

y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

inputs = Input((28, 28, 1))

c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
cc1 = concatenate([c1, inputs], axis=3)
p1 = MaxPooling2D((2, 2))(cc1)

c21 = Conv2D(8, (1, 1), activation='relu', padding='same')(p1)
c22 = Conv2D(8, (3, 3), activation='relu', padding='same')(p1)
c23 = Conv2D(8, (5, 5), activation='relu', padding='same')(p1)
cc2 = concatenate([c21, c22, c23], axis=3)

flat = Flatten()(cc2)
d1 = Dense(32, activation='relu')(flat)
d2 = Dense(10, activation='softmax')(d1)

model = Model(inputs=inputs, outputs=d2)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

onnx_model, _ = tf2onnx.convert.from_keras(model)

onnx.save(onnx_model, 'mnist_cnn.onnx')

# keras2onnx.convert_keras(model, name="CNN_ROOT")

