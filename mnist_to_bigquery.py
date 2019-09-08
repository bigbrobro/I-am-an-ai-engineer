import os
import tensorflow as tf
import pandas as pd
import numpy as np
from utils import bigquery_config, phase

_project = bigquery_config[phase]['project']
_dataset = bigquery_config[phase]['dataset']


def extract_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    return x_train, y_train, x_test, y_test


def data_flatten(x_train, y_train, x_test, y_test):
    with open('data/train.csv', 'w') as output:
        for num, (image, label) in enumerate(zip(x_train, y_train)):
            img = ",".join(map(str, image.flatten().tolist()))
            output.write(f"{num}|{img}|{label}|train")
            output.write('\n')

    with open('data/test.csv', 'w') as output:
        for num, (image, label) in enumerate(zip(x_test, y_test)):
            img = ",".join(map(str, image.flatten().tolist()))
            output.write(f'{num}|{img}|{label}|test')
            output.write('\n')


def read_and_concat():
    train_df = pd.read_csv("./data/train.csv", sep='|', names=['num', 'image', 'label', 'type'])
    test_df = pd.read_csv("./data/test.csv", sep='|', names=['num', 'image', 'label', 'type'])
    concat_df = train_df.append(test_df)
    return concat_df


def run_mnist_to_bigquery():
    x_train, y_train, x_test, y_test = extract_data()
    data_flatten(x_train, y_train, x_test, y_test)
    concat_df = read_and_concat()
    concat_df.to_gbq(destination_table=f"{_dataset}.data", project_id=_project, if_exists='replace')
    print("mnist to bigquery job is end")


def download_data_from_bigquery():
    if not os.path.exists('data/x_test.npy'):
        data_df = pd.read_gbq("SELECT image, label, type FROM `geultto.mnist.data` ",
                              dialect='standard', project_id=_project)
        train_df = data_df[data_df['type'] == 'train'].reset_index(drop=True)
        test_df = data_df[data_df['type'] == 'test'].reset_index(drop=True)

        x_train = []
        y_train = []
        for i in range(len(train_df)):
            image = np.asarray(train_df['image'][i].split(',')).astype(np.uint8)
            label = np.asarray(train_df['label'][i]).astype(np.uint8)

            x_train.append(image)
            y_train.append(label)

        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)
        np.save('data/x_train', x_train)
        np.save('data/y_train', y_train)

        x_test = []
        y_test = []
        for i in range(len(test_df)):
            image = np.asarray(test_df['image'][i].split(',')).astype(np.uint8)
            label = np.asarray(test_df['label'][i]).astype(np.uint8)

            x_test.append(image)
            y_test.append(label)

        x_test = np.asarray(x_test)
        y_test = np.asarray(y_test)
        np.save('data/x_test', x_test)
        np.save('data/y_test', y_test)

