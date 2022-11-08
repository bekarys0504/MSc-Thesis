import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import math
import random
import logging
import click
from dotenv import find_dotenv, load_dotenv
from omegaconf import OmegaConf
from pathlib import Path

@click.command()

def main():
    path = r'C:\Users\bcilab02\Documents\beka\thesis\MSc-Thesis\data\processed\deep_learning_data\train\Depressed/'
    X = []
    labels = []
    for i in os.listdir(path):
        data = np.load(path+i)
        X.append(data)
        labels.append(0)

    path = r'C:\Users\bcilab02\Documents\beka\thesis\MSc-Thesis\data\processed\deep_learning_data\train\Healthy/'
    for i in os.listdir(path):
        data = np.load(path+i)
        X.append(data)
        labels.append(1)

    y = labels
    temp = list(zip(X, y))
    random.shuffle(temp)
    X, y = zip(*temp)

    X, y = list(X), list(y)
    X = np.array(X)
    y = np.array(y)
    X = X[..., np.newaxis]

    X_train      = X[0:1500,]
    Y_train      = y[0:1500]
    X_validate   = X[1500:,]
    Y_validate   = y[1500:]

    model=models.Sequential()
    model.add(layers.Conv2D(16,(16,13),activation='elu',input_shape=(3000,31,1), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.AveragePooling2D((6,6)))
    model.add(layers.Dropout(rate=0.25))

    model.add(layers.Conv2D(32,(12,5), activation='elu', padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.AveragePooling2D((4,4)))
    model.add(layers.Dropout(rate=0.25))

    model.add(layers.Conv2D(64,(10,1), activation='elu', padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.AveragePooling2D((3,1)))
    model.add(layers.Dropout(rate=0.25))

    model.add(layers.Conv2D(128,(8,1), activation='elu', padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.AveragePooling2D((2 ,1)))
    model.add(layers.Dropout(rate=0.25))

    model.add(layers.Conv2D(256,(6,1), activation='elu', padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.AveragePooling2D((2,1)))
    model.add(layers.Dropout(rate=0.25))

    model.add(layers.Flatten())
    #model.add(layers.Dense(50, activation='elu'))
    model.add(layers.Dense(2))
    model.summary()


    EPOCHS = 200
    BATCH_SIZE = 32
    step_per_epoch = math.ceil(X_train.shape[0]/BATCH_SIZE)
    STEP_TOTAL = step_per_epoch * EPOCHS
    INIT_LEARNING_RATE =  0.001

    lr_schedule=CosineDecay(INIT_LEARNING_RATE,STEP_TOTAL,
                                                        alpha=0.0,
                                                        name=None)
    optimizer =Adam(learning_rate=lr_schedule)

    model.compile(optimizer=optimizer,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']
                )


    checkpoint_filepath = './models/weights/best.hdf5'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    history = model.fit(X_train,Y_train,
                        epochs=EPOCHS,
                        validation_data=(X_validate,Y_validate),
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        callbacks=[model_checkpoint_callback])

    test_predictions = model.predict(x = X_validate)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    config = OmegaConf.load('./config/config.yaml')
    main()