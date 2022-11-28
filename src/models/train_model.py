import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras import layers, models
import numpy as np
import math
import random
import logging
import click
from dotenv import find_dotenv, load_dotenv
from omegaconf import OmegaConf
from pathlib import Path
import wandb
from tensorflow.keras.layers import LeakyReLU
from sklearn.metrics import accuracy_score
from tensorflow.keras import regularizers
from wandb.keras import WandbCallback
import pickle


wandb.init(project="DeepEEG", entity="bekarys")
config = OmegaConf.load('./config/config.yaml')

# set random seeds for reproducability
tf.random.set_seed(config['random_seed'])
np.random.seed(config['random_seed'])
random.seed(config['random_seed'])
os.environ['PYTHONHASHSEED'] = '0'
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

@click.command()

def main():

    path = r'.\data\processed\deep_learning_data\no_overlap_data/Depressed/'
    X = []
    labels = []
    for i in os.listdir(path):
        data = np.load(path+i)
        X.append(data)
        labels.append(1)

    path = r'.\data\processed\deep_learning_data\no_overlap_data/Healthy/'
    for i in os.listdir(path):
        data = np.load(path+i)
        X.append(data)
        labels.append(0)

    y = labels
    temp = list(zip(X, y))
    random.shuffle(temp)
    X, y = zip(*temp)

    X, y = list(X), list(y)
    X = np.array(X)
    y = np.array(y)

    X = X[..., np.newaxis]

    X_train      = X[0:5700,]
    Y_train      = y[0:5700]
    X_val   = X[5700:6700,]
    Y_val   = y[5700:6700]
    X_test   = X[6700:,]
    Y_test   = y[6700:]

    # print class percentages
    neg, pos = np.bincount(Y_train)
    total = neg + pos
    print('Train: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
        total, pos, 100 * pos / total))
    
    neg, pos = np.bincount(Y_val)
    total = neg + pos
    print('Validation: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
        total, pos, 100 * pos / total))

    neg, pos = np.bincount(Y_test)
    total = neg + pos
    print('Test: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
        total, pos, 100 * pos / total))

    neg, pos = np.bincount(y)
    total = neg + pos
    print('Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
        total, pos, 100 * pos / total))
    
    initial_bias = np.log([pos/neg])

    # Scaling by total/2 helps keep the loss to a similar magnitude.
    # The sum of the weights of all examples stays the same.
    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)

    class_weight = {0: weight_for_0, 1: weight_for_1}

    print('Weight for class 0: {:.2f}'.format(weight_for_0))
    print('Weight for class 1: {:.2f}'.format(weight_for_1))

    EPOCHS = config['deep_learning_hp']['epochs']
    BATCH_SIZE = config['deep_learning_hp']['batch_size']
    INIT_LEARNING_RATE =  config['deep_learning_hp']['lr']
    step_per_epoch = math.ceil(X_train.shape[0]/BATCH_SIZE)
    STEP_TOTAL = step_per_epoch * EPOCHS
    DROPOUT_RATE =  config['deep_learning_hp']['dropout_rate']

    wandb.config.update({
    "learning_rate": INIT_LEARNING_RATE,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "dropout_rate": DROPOUT_RATE,
    "pooling": 'max_pooling'
    })

    lr_schedule=CosineDecay(INIT_LEARNING_RATE, STEP_TOTAL,
                                                alpha=0.0,
                                                name=None)
    optimizer =Adam(learning_rate=lr_schedule)
    
    model = get_model(input_shape=(X_train.shape[1],X_train.shape[2],1),dropout_rate=DROPOUT_RATE, output_bias=initial_bias)
    model.compile(optimizer=optimizer,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']
                )

    model_checkpoint_callback, model_path = get_callback()

    # save test files
    np.save(model_path+'/x_test.npy', X_test)
    np.save(model_path+'/y_test.npy', Y_test)

    history = model.fit(X_train,Y_train,
                        epochs=EPOCHS,
                        validation_data=(X_val,Y_val),
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        callbacks=[model_checkpoint_callback, WandbCallback()], 
                        class_weight=class_weight)

    with open(model_path+'/trainHistoryDict', 'wb') as file_pi:
        print('saving logs in ', model_path+'/trainHistoryDict')
        pickle.dump(history.history, file_pi)

    test_predictions = model.predict(x = X_test)
    predictions = np.argmax(test_predictions, axis=1)
    print(predictions)
    print(Y_test)
    print('Test accuracy: ',accuracy_score(Y_test, predictions))

def get_data(data_path):
    X = []
    labels = []

    for i in os.listdir(data_path+'Depressed/'):
        data = np.load(data_path+'Depressed/'+i)
        if not np.any(np.isnan(data)):
            X.append(data)
            labels.append(1)


    for i in os.listdir(data_path+'Healthy/'):
        data = np.load(data_path+'Healthy/'+i)

        if not np.any(np.isnan(data)):
            X.append(data)
            labels.append(0)

    X = np.array(X)
    labels = np.array(labels)
    X = X[..., np.newaxis]
    return X, labels

def get_callback():
    directory = './models/weights/exp'
    for i in range(1, 100):
        if not os.path.exists(directory+str(i)):
            os.makedirs(directory+str(i))
            break

    checkpoint_filepath = directory+str(i)+'/best.hdf5'
    
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)
    return model_checkpoint_callback, directory+str(i)


def get_model(input_shape=(500,31,1), dropout_rate=0.25, output_bias=0):

    model=models.Sequential()
    model.add(layers.Conv2D(3,(11,7), input_shape=input_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.1))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(rate=dropout_rate))

    model.add(layers.Conv2D(5,(11,7), padding="same"))
    model.add(LeakyReLU(alpha=0.1))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(rate=dropout_rate))

    model.add(layers.Conv2D(5,(11,7), padding="same"))
    model.add(LeakyReLU(alpha=0.1))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 1)))
    model.add(layers.Dropout(rate=dropout_rate))
    
    '''   
    model.add(layers.Conv2D(10,(11,7), padding="same"))
    model.add(LeakyReLU(alpha=0.1))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,1)))
    model.add(layers.Dropout(rate=dropout_rate))

    model.add(layers.Conv2D(16,(11,7), padding="same"))
    model.add(LeakyReLU(alpha=0.1))
    model.add(layers.BatchNormalization())
    model.add(layers.AveragePooling2D((2,1)))
    model.add(layers.Dropout(rate=dropout_rate))
    '''
    model.add(layers.Flatten())
    #model.add(layers.Dense(1024))
    #model.add(LeakyReLU(alpha=0.1))
    #model.add(layers.Dropout(rate=dropout_rate))
    model.add(layers.Dense(256))
    model.add(LeakyReLU(alpha=0.1))
    model.add(layers.Dropout(rate=dropout_rate))
    model.add(layers.Dense(2, activation='softmax', bias_initializer=tf.keras.initializers.Constant(output_bias)))

    return model

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    main()