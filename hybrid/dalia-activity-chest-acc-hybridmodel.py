import os

os.environ['PYTHONHASHSEED'] = '0'
import numpy as np

np.random.seed(1)
import random as rn

rn.seed(1)
import tensorflow as tf

tf.random.set_seed(1)
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.keras import callbacks
from tensorflow.keras import optimizers
import pandas as pd
import argparse
import seaborn as sns
import csv
from numpy import save, load
import time
from scipy import stats
from sklearn import metrics
from sklearn.metrics import classification_report, roc_auc_score
from sklearn import preprocessing
from scipy.stats import dirichlet
from matplotlib import pyplot as plt
import math
import sys
import glob

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

# Set some standard parameters upfront
pd.options.display.float_format = '{:.1f}'.format
plt.style.use('ggplot')

LABELS = [1, 2, 3, 4, 5, 6, 7, 8]
TIME_PERIODS = 40
STEP_DISTANCE = 40
N_FEATURES = 3
RESAMPLE = 175
num_classes = len(LABELS)
print(num_classes)


def write_list_to_file(guest_list, filename):
    """Write the list to csv file."""
    with open(filename, "w") as output:
        writer = csv.writer(output, delimiter=',', lineterminator='\n')
        for row in enumerate(guest_list):
            writer.writerows([row])


def read_data(cwd, filepath):
    print(filepath)
    os.chdir(cwd + "/" + filepath)
    extension = 'pkl'
    all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
    print(all_filenames)
    frames = []
    label = []
    ECG = []
    EDA = []
    BVP = []
    EMG = []
    RESP = []
    ACC_X = []
    ACC_Y = []
    ACC_Z = []
    for f in all_filenames:
        print(f)
        df = pd.read_pickle(f)
        list_label = df['activity'].tolist()
        np.set_printoptions(threshold=sys.maxsize)
        #print(list_label)
        list_chest_ACC = df['signal']['chest']['ACC'].tolist()
        print('length of activity: ', len(list_label))
        for i in range(0, int(len(list_chest_ACC) / RESAMPLE)):
            ACC_X.append(list_chest_ACC[i * RESAMPLE][0])
            ACC_Y.append(list_chest_ACC[i * RESAMPLE][1])
            ACC_Z.append(list_chest_ACC[i * RESAMPLE][2])
        count_label = [0, 0, 0, 0, 0, 0, 0, 0]
        for j in range(0, int(len(list_label))):
            if list_label[j][0] == 1:
                count_label[0] = count_label[0] + 1
            if list_label[j][0] == 2:
                count_label[1] = count_label[1] + 1
            if list_label[j][0] == 3:
                count_label[2] = count_label[2] + 1
            if list_label[j][0] == 4:
                count_label[3] = count_label[3] + 1
            if list_label[j][0] == 5:
                count_label[4] = count_label[4] + 1
            if list_label[j][0] == 6:
                count_label[5] = count_label[5] + 1
            if list_label[j][0] == 7:
                count_label[6] = count_label[6] + 1
            if list_label[j][0] == 8:
                count_label[7] = count_label[7] + 1
            label.append(list_label[j][0])
        print(count_label)
        print('length of ACC-X: ', len(ACC_X))
        print('length of ACC-Y: ', len(ACC_Y))
        print('length of ACC-Z: ', len(ACC_Z))
        print('length of activity: ', len(label))
        df_fn = pd.DataFrame(list(zip(label, ACC_X, ACC_Y, ACC_Z)),
                             columns=['label', 'ACC-X', 'ACC-Y', 'ACC-Z'])
        frames.append(df_fn)
    df_frames = pd.concat(frames)
    os.chdir(cwd)
    print('finished loading data...')
    return df_frames


def create_segments_and_labels(df, time_steps, step):
    print('=====starting to segment====')
    segments = []
    labels = []
    count_label = [0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(0, len(df) - time_steps, step):
        acc_x = df['ACC-X'].values[i: i + time_steps]
        acc_y = df['ACC-Y'].values[i: i + time_steps]
        acc_z = df['ACC-Z'].values[i: i + time_steps]
        label = stats.mode(df['label'][i: i + time_steps])[0]
        if label in LABELS:
            segments.append([acc_x, acc_y, acc_z])
            labels.append(label)
        for k in range(len(LABELS)):
            if label == LABELS[k]:
                count_label[k] = count_label[k] + 1

    max = 0
    sum = 0
    for j in range(num_classes):
        sum += count_label[j]
        if (count_label[j] > max):
            max = count_label[j]
        print(LABELS[j], count_label[j])
    class_weighted = [0, 0, 0, 0, 0, 0, 0, 0]
    ratio_class = [0, 0, 0, 0, 0, 0, 0, 0]
    for j in range(num_classes):
        if (count_label[j] != 0):
            class_weighted[j] = round(1 - (count_label[j] / sum), 2)
            ratio_class[j] = int(max / count_label[j])
    print('weighted classes:', class_weighted)
    print('ratio class:', ratio_class)
    reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, time_steps, N_FEATURES)
    labels = np.asarray(labels)
    print('=====end segment====')
    return reshaped_segments, labels, class_weighted, ratio_class


def feature_normalize(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    return (dataset - mu) / sigma


def convert_to_float(x):
    try:
        return np.float(x)
    except:
        return np.nan


def show_basic_dataframe_info(dataframe):
    print('Number of columns in the dataframe: %i' % (dataframe.shape[1]))
    print('Number of rows in the dataframe: %i\n' % (dataframe.shape[0]))


def show_confusion_matrix(validations, predictions):
    matrix = metrics.confusion_matrix(validations, predictions)
    plt.figure(figsize=(6, 6))
    sns.heatmap(matrix,
                cmap='coolwarm',
                linecolor='white',
                linewidths=1,
                xticklabels=LABELS,
                yticklabels=LABELS,
                annot=True,
                fmt='d')
    plt.title('Confusion Matrix ACTIVITY')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


def roc_auc_score_multiclass(actual_class, pred_class, average="macro"):
    unique_class = set(actual_class)
    roc_auc_dict = {}
    for per_class in unique_class:
        other_class = [x for x in unique_class if x != per_class]
        new_actual_class = [0 if x in other_class else 1 for x in actual_class]
        new_pred_class = [0 if x in other_class else 1 for x in pred_class]
        roc_auc = roc_auc_score(new_actual_class, new_pred_class, average=average)
        roc_auc_dict[per_class] = roc_auc
    return roc_auc_dict


def expected_calibration_error(y_true, y_pred, list_ratio_class):
    pred_y = np.argmax(y_pred, axis=1)
    eval = (pred_y == y_true)
    list_false_eval = np.where(eval == False)
    beta = dirichlet(list_ratio_class, seed=0)
    beta = beta.mean()
    for i in list_false_eval:
        y_pred[i] = np.multiply(y_pred[i], beta)
    return y_pred


def count_labels(labels):
    count_label = [0, 0, 0, 0, 0]
    for label in labels:
        if label == "Desk Work":
            count_label[0] += 1
        elif label == "Discussion":
            count_label[1] += 1
        elif label == "Lunch":
            count_label[2] += 1
        elif label == "Presentation":
            count_label[3] += 1
        elif label == "Walking":
            count_label[4] += 1
    print(count_label)
    return count_label


def data_preprocessing(dataset_path, time_steps, step, N_FEATURES, resample, LABELS):
    cwd = os.getcwd()
    df = read_data(cwd, dataset_path)
    show_basic_dataframe_info(df)
    df['ACC-X'] = feature_normalize(df['ACC-X'])
    df['ACC-Y'] = feature_normalize(df['ACC-Y'])
    df['ACC-Z'] = feature_normalize(df['ACC-Z'])
    df = df.dropna()
    segments, labels, class_weighted, ratio_class = create_segments_and_labels(df, time_steps, step)
    labels = labels.astype(np.int32)
    one_hot_labels = to_categorical(labels - 1, num_classes=len(LABELS))
    print("Processed data shape: ", segments.shape)
    return segments, one_hot_labels, class_weighted, ratio_class


def build_cnn_lstm_model(input_shape, num_classes):
    model = keras.Sequential()

    # Add a 1D Convolutional layer
    model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Dropout(0.5))

    # Add an LSTM layer
    model.add(layers.LSTM(100, return_sequences=True))
    model.add(layers.LSTM(100))
    model.add(layers.Dropout(0.5))

    # Add a Dense layer with softmax activation
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    # Print the model summary
    print(model.summary())
    return model


def train_model(train_data, train_labels, test_data, test_labels, input_shape, num_classes):
    model = build_cnn_lstm_model(input_shape, num_classes)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(),
                  metrics=['accuracy'])

    callback = callbacks.EarlyStopping(monitor='loss', patience=3)
    history = model.fit(train_data,
                        train_labels,
                        epochs=4,
                        batch_size=64,
                     #   validation_split=0.2,
                        callbacks=[callback],
                        verbose=1)

    score = model.evaluate(test_data, test_labels, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    return model, history


def main():
    # Parse arguments
    argument_parser = argparse.ArgumentParser(
        description="CLI for training and testing Sequential Neural Network Model")
    argument_parser.add_argument("--train_file", type=str,
                                 default="\DALIA\\train", help="Train file (CSV). Required for training.")
    argument_parser.add_argument("--validation_file", type=str, help="Validation file (CSV). Required for validation.")
    argument_parser.add_argument("--test_file", type=str,
                                 default="\DALIA\\test", help="Test file (CSV). Required for testing.")
    args = argument_parser.parse_args()

    # Load and preprocess data
    train_segments, train_one_hot_labels, train_class_weighted, train_ratio_class = data_preprocessing(args.train_file,
                                                                                                       TIME_PERIODS,
                                                                                                       STEP_DISTANCE,
                                                                                                       N_FEATURES,
                                                                                                       RESAMPLE,
                                                                                                       LABELS)
    test_segments, test_one_hot_labels, _, _ = data_preprocessing(args.test_file, TIME_PERIODS,
                                                                  STEP_DISTANCE, N_FEATURES, RESAMPLE,
                                                                  LABELS)


    # Train the model
    input_shape = (TIME_PERIODS, N_FEATURES)
    model, history = train_model(train_segments, train_one_hot_labels, test_segments, test_one_hot_labels, input_shape, num_classes)

    # ======Print confusion matrix for training data========
    y_pred_train = model.predict(train_segments)
    # print('y_pred_train: ', y_pred_train)
    # Take the class with the highest probability from the train_0_6 predictions
    max_y_pred_train = np.argmax(y_pred_train, axis=1)
    max_train = np.argmax(train_one_hot_labels, axis=1)
    print(classification_report(max_train, max_y_pred_train))

    # Make predictions
    y_pred = model.predict(test_segments)
    pred_max_y = np.argmax(y_pred, axis=1)
    pred_max_y = pred_max_y + 1
    validation_max_y = np.argmax(test_one_hot_labels, axis=1)
    validation_max_y = validation_max_y + 1

    # Show confusion matrix
    show_confusion_matrix(validation_max_y, pred_max_y)

    # Show classification reports
    print(classification_report(validation_max_y, pred_max_y, target_names=[str(i) for i in LABELS]))

    # Calculate ROC AUC score
    roc_auc = roc_auc_score_multiclass(validation_max_y, pred_max_y)
    print("ROC AUC Score: ", roc_auc)

    # Calculate Expected Calibration Error
    ece = expected_calibration_error(validation_max_y, y_pred, train_ratio_class)
    #print("Expected Calibration Error: ", ece)


if __name__ == "__main__":
    main()
