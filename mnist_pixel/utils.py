import numpy as np
import tensorflow as tf 
tf.enable_eager_execution()
from keras.utils import to_categorical

def data_generator(permute=False):
    # input image dimensions
    img_rows, img_cols = 28, 28
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, img_rows * img_cols, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, img_rows * img_cols, 1).astype('float32') / 255.

    y_train = y_train.astype('int32')
    y_test = y_test.astype('int32')
    # shape=(60000, 784, 1) (60000, 10),  (10000, 784, 1) (10000, 10)

    # Sequential MNIST
    if not permute:
        return (x_train, y_train), (x_test, y_test)
    else:
        # Permuted Sequential MNIST
        print("Using Permuted Sequential MNIST...")
        random_order = np.arange(img_rows*img_cols)
        np.random.shuffle(random_order)
        x_train = x_train[:, random_order, :]
        x_test = x_test[:, random_order, :]
        return (x_train, y_train), (x_test, y_test)
    print("ERROR in data_generator!")

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = data_generator()
    print(y_train.dtype)
