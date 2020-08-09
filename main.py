from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

from pairwise_compatibility import polyvore_dataset, DataGenerator
from utils import Config
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow import keras
from tensorflow.keras.models import load_model

if __name__ == '__main__':

    #     data generator

    dataset = polyvore_dataset()
    transforms = dataset.get_data_transforms()
    X_train, X_test, y_train, y_test, n_classes = dataset.create_dataset()

    if Config['debug']:
        train_set = (X_train[:500], y_train[:500], transforms['train'])
        test_set = (X_test[:500], y_test[:500], transforms['test'])
        dataset_size = {'train': 500, 'test': 500}
    else:
        train_set = (X_train[:100000], y_train[:100000], transforms['train'])
        test_set = (X_test[:20000], y_test[:20000], transforms['test'])
        dataset_size = {'train': 100000, 'test': 20000}

    params = {'batch_size': Config['batch_size'],
              'n_classes': n_classes,
              'shuffle': True
              }

    train_generator = DataGenerator(train_set, dataset_size, params)
    test_generator = DataGenerator(test_set, dataset_size, params)

    # Use GPU

    model = load_model('Compatibility-model.hdf5')
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    mc = keras.callbacks.ModelCheckpoint('Saved_epoch.hdf5', save_weights_only=False)
    model.summary()

    # training
    results = model.fit(train_generator,
                        validation_data=test_generator,
                        epochs=Config['num_epochs'], callbacks=[mc]
                        )

    loss = results.history['loss']
    val_loss = results.history['val_loss']
    acc = results.history['accuracy']
    val_acc = results.history['val_accuracy']

    epochs = np.arange(len(loss))

    plt.figure()
    plt.plot(epochs, acc, label='acc')
    plt.plot(epochs, val_acc, label='val_acc')
    plt.xlabel('epochs')
    plt.ylabel('Accuracy')
    plt.title('Compatibility_Model')
    plt.legend()
    plt.savefig('Compatibility_learning_acc.png', dpi=256)