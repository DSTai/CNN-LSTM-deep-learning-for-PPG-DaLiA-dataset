import os
os.environ['PYTHONHASHSEED']= '0'
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
#%matplotlib inline
import pandas as pd
import argparse
import seaborn as sns
import csv
from numpy import save,load
import time
#import coremltools
from scipy import stats
from sklearn import metrics
from sklearn.metrics import classification_report, roc_auc_score
from sklearn import preprocessing
from scipy.stats import dirichlet
from matplotlib import pyplot as plt
import math
import sys
import glob
#import keras
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Flatten, Reshape, LSTM
#from keras.utils import np_utils
#from keras.callbacks import ModelCheckpoint
#from keras import optimizers


import tensorflow as tf
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

#=========================================#
#                                         #
#             BEGIN                       #
#                                         #
#=========================================#
# Set some standard parameters upfront
pd.options.display.float_format = '{:.1f}'.format
#sns.set() # Default seaborn look and feel
plt.style.use('ggplot')
#print('keras version ', keras.__version__)
# Same labels will be reused throughout the program

LABELS = [1, 2, 3, 4, 5, 6, 7, 8]
#LABELS = ["Desk Work","Eating/Drinking","Movement","Sport","unknown"]
# The number of steps within one time segment
TIME_PERIODS = 40
# The steps to take from one segment to the next; if this value is equal to
# TIME_PERIODS, then there is no overlap between the segments
STEP_DISTANCE = 40
N_FEATURES = 3
RESAMPLE = 8
num_classes = len(LABELS)
print(num_classes)

def write_list_to_file(guest_list, filename):
    """Write the list to csv file."""
    with open(filename, "w") as output:
	    writer = csv.writer(output, delimiter = ',', lineterminator='\n')
	    for row in enumerate(guest_list):
	    	writer.writerows([row])

def read_data(cwd, filepath):
    #print('====loading data====')
    #df = pd.concat(map(pd.read_pickle, glob.glob(os.path.join('', filepath))))
    # Parse paths
    print(filepath)
    os.chdir(cwd + "/"+  filepath)
    extension = 'pkl'
    all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
    print(all_filenames)
    # combine all files in the list
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
        #print(df)
        list_label = df['activity'].tolist()
        np.set_printoptions(threshold=sys.maxsize)
        #print(list_label)
        #list_chest_ECG = df['signal']['chest']['ECG'].tolist()
        #list_chest_EDA = df['signal']['chest']['EDA'].tolist()
        #list_chest_EMG = df['signal']['chest']['EMG'].tolist()
        #list_chest_RESP = df['signal']['chest']['Resp'].tolist()
        #list_chest_ACC = df['signal']['chest']['ACC'].tolist()
        list_wrist_ACC = df['signal']['wrist']['ACC'].tolist()
        #list_wrist_EDA = df['signal']['wrist']['EDA'].tolist()
        #list_wrist_BVP = df['signal']['wrist']['BVP'].tolist()
        #print('length of BVP: ', len(list_wrist_BVP))
        #print('length of EDA: ', len(list_wrist_EDA))
        #print('length of EDA: ', len(list_wrist_EDA))
        print('length of activity: ', len(list_label))
        for i in range(0, int(len(list_wrist_ACC)/RESAMPLE)):
            #ECG.append(list_chest_ECG[i][0])
            #RESP.append(list_chest_RESP[i*RESAMPLE][0])
            ACC_X.append(list_wrist_ACC[i * RESAMPLE][0])
            ACC_Y.append(list_wrist_ACC[i * RESAMPLE][1])
            ACC_Z.append(list_wrist_ACC[i * RESAMPLE][2])
            #EDA.append(list_chest_EDA[i][0])
            #EMG.append(list_chest_EMG[i][0])
            #EDA.append(list_wrist_EDA[i][0])
            #BVP.append(list_wrist_BVP[i][0])
        #print(int(len(list_label)))
        #print(list_label)
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
            #if list_label[j][0] == 8:
            #    count_label[8] = count_label[8] + 1
            label.append(list_label[j][0])
        print(count_label)
        print('length of ACC-X: ', len(ACC_X))
        print('length of ACC-Y: ', len(ACC_Y))
        print('length of ACC-Z: ', len(ACC_Z))
        print('length of activity: ', len(label))
        #plt.plot(ECG)
        #plt.show()
        df_fn = pd.DataFrame(list(zip(label,
                                      #ECG,
                                      #RESP
                                      ACC_X,
                                      ACC_Y,
                                      ACC_Z,
                                      #EDA
                                      #BVP
                                      #EMG
                                      )),
                             columns=['label',
                                      #'ECG',
                                      'ACC-X',
                                      'ACC-Y',
                                      'ACC-Z'
                                      #'EDA'
                                      #'BVP'
                                      #'EMG'
                                      ])
        frames.append(df_fn)
    df_frames = pd.concat(frames)

    #df = pd.read_pickle(filepath)
    #
    #print(df['label'])
    #print(df['signal']['chest']['ECG'])
    #print(df['signal']['chest']['EDA'])
    #print(df['signal']['chest']['EMG'])


    #print(len(df['signal']['chest']['RESP']))
    #print(len(df['signal']['chest']['TEMP']))
    #print(df['label'])

    #df = pd.read_csv(filepath, skip_blank_lines=True, na_filter=True).dropna()
    #df = df.drop(labels=['timestamp'], axis=1)
    #df.dropna(how='any', inplace=True)
    #cols_to_norm = ['mean_x_p', 'mean_y_p', 'mean_z_p', 'var_x_p', 'var_y_p', 'var_z_p',
    #                'mean_x_w', 'mean_y_w', 'mean_z_w', 'var_x_w', 'var_y_w', 'var_z_w']
    #df[cols_to_norm] = df[cols_to_norm].apply(lambda x: (x - x.mean()) / (x.max() - x.min()))
    # Round numbers
    #df = df.round({'attr_x': 4, 'attr_y': 4, 'attr_z': 4})
    print('finished loading data...')
    return df_frames

def create_segments_and_labels(df, time_steps, step):
    print('=====starting to segment====')
    # x, y, z acceleration as features

    # Number of steps to advance in each iteration (for me, it should always
    # be equal to the time_steps in order to have no overlap between segments)
    # step = time_steps
    segments = []
    labels = []
    count_label = [0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(0, len(df) - time_steps, step):
        #x = df['ECG'].values[i: i + time_steps]
        #eda = df['EDA'].values[i: i + time_steps]
        #resp = df['RESP'].values[i: i + time_steps]
        acc_x = df['ACC-X'].values[i: i + time_steps]
        acc_y = df['ACC-Y'].values[i: i + time_steps]
        acc_z = df['ACC-Z'].values[i: i + time_steps]
        #bvp = df['BVP'].values[i: i + time_steps]
        #z = df['EMG'].values[i: i + time_steps]
        # Retrieve the most often used label in this segment
        #print(df['label'][i: i + time_steps])[0][0]
        #print(df['label'][i: i + time_steps])[0][i + time_steps]
        label = stats.mode(df['label'][i: i + time_steps])[0]#[0]
        if label in LABELS:
            segments.append([#x,
                             #eda
                             #resp
                             acc_x,
                             acc_y,
                             acc_z,
                             #bvp
                             #z
                            ])
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
    # sample_weighted = [0,0,0,0]
    for j in range(num_classes):
        if(count_label[j]!=0):
            class_weighted[j] = round(1 - (count_label[j] / sum), 2)
            ratio_class[j] = int(max / count_label[j])
    print('weighted classes:', class_weighted)
    print('ratio class:', ratio_class)
    # Bring the segments into a better shape
    #print(segments)
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
    # Shape and how many rows and columns
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

def roc_auc_score_multiclass(actual_class, pred_class, average = "macro"):

  #creating a set of all the unique classes using the actual class list
  unique_class = set(actual_class)
  roc_auc_dict = {}
  for per_class in unique_class:
    #creating a list of all the classes except the current class
    other_class = [x for x in unique_class if x != per_class]

    #marking the current class as 1 and all other classes as 0
    new_actual_class = [0 if x in other_class else 1 for x in actual_class]
    new_pred_class = [0 if x in other_class else 1 for x in pred_class]

    #using the sklearn metrics method to calculate the roc_auc_score
    roc_auc = roc_auc_score(new_actual_class, new_pred_class, average = average)
    roc_auc_dict[per_class] = roc_auc

  return roc_auc_dict

def expected_calibration_error(y_true, y_pred, list_ratio_class):
    #print('y_pred before calibration: ', y_pred)
    pred_y = np.argmax(y_pred, axis=1)
    eval = (pred_y == y_true)
    #print('evaluate y_pred and y_true:', eval)
    list_false_eval = np.where(eval == False)
    #print('list_false_eval:', list_false_eval)
    #compute the minimized of Kullback-leibler divergence
    #print('list_ratio_class: ', list_ratio_class)
    beta = dirichlet(list_ratio_class, seed=0)
    beta = beta.mean()
    #print('dirichlet list_ratio_class: ', beta)
    for i in list_false_eval:
        y_pred[i] = np.multiply(y_pred[i], beta)
    #print('y_pred after calibration: ', y_pred)
    return y_pred

def count_labels(labels):
    count_label = [0, 0, 0, 0, 0]
    for label in labels:
        if label == "Desk Work":
            count_label[0] += 1
        elif label == "Eating/Drinking":
            count_label[1] += 1
        #elif label == "Housework":
        #    count_label[2] += 1
        #elif label == "Meal preparation":
        #    count_label[3] += 1
        elif label == "Movement":
            count_label[2] += 1
        #elif label == "Personal Grooming":
        #    count_label[3] += 1
        #elif label == "Relaxing":
        #    count_label[4] += 1
        #elif label == "Shopping":
        #    count_label[7] += 1
        #elif label == "Socializing":
        #    count_label[5] += 1
        elif label == "Sport":
            count_label[3] += 1
        #elif label == "Transportation":
        #    count_label[7] += 1
        elif label == "unknown":
            count_label[4] += 1

    max = 0
    sum = 0
    for j in range(num_classes):
        sum += count_label[j]
        if count_label[j] > max:
            max = count_label[j]
        print(LABELS[j], count_label[j])

    class_weighted = [0, 0, 0, 0, 0]
    ratio_class = [0, 0, 0, 0, 0]
    for j in range(num_classes):
        if(count_label[j] != 0):
          class_weighted[j] = round(1 - (count_label[j] / sum), 2)
          ratio_class[j] = int(max / count_label[j])
    print('weighted classes:', class_weighted)
    print('ratio class:', ratio_class)
    return count_label, class_weighted, ratio_class

def calculate_weighted_entropy(array, list_weighted_classes):
    list_entropy = []
    total_rows = array.shape[1]
    # print "total_rows", total_rows
    for i in range(total_rows):
        entropy = 0
        sum_ratio_classes = 0
        total_columns = array.shape[0]
        # print "total_columns",total_columns
        for l in range(total_columns):
            sum_ratio_classes += list_weighted_classes[l] * array[l][i]
        for j in range(total_columns):
            if (array[j][i] != 0):
                x = float(array[j][i] * list_weighted_classes[j]) / sum_ratio_classes
                entropy += x * math.log(x)
                #entropy += (array[j][i]* math.log(array[j][i]))* list_ratio_classes[j]
        list_entropy.append(-entropy)

    return list_entropy

def concatenate(train_xs, train_ys, validation_xs, validation_ys):
    train_xs = np.concatenate((train_xs, validation_xs))
    train_ys = np.concatenate((train_ys, validation_ys))
    return train_xs, train_ys


argument_parser = argparse.ArgumentParser(description="CLI for training and testing Sequential Neural Network Model")
argument_parser.add_argument("--train_file", type=str,
                             default="\DALIA\\train",help="Train file (CSV). Required for training.")
argument_parser.add_argument("--validation_file", type=str, help="Validation file (CSV). Required for validation.")
argument_parser.add_argument("--test_file", type=str,
                             default="\DALIA\\test",help="Test file (CSV). Required for testing.")
args = argument_parser.parse_args()

# Load data set containing all the data from csv
# Define column name of the label vector
#LABEL = 'ActivityEncoded'
# Transform the labels from String to Integer via LabelEncoder
le = preprocessing.LabelEncoder()
# Add a s2.txt column to the existing DataFrame with the encoded values
start_time = time.time()
cwd = os.getcwd()
train = read_data(cwd, args.train_file)
x_train, y_train, class_weight_train, ratio_class_train = create_segments_and_labels(train, TIME_PERIODS, STEP_DISTANCE)
#print("=======saving train_0_6 data==============")
#save('save_segmented_data/x_train_0_2_activity_sub2_50_50.npy', x_train)
#save('save_segmented_data/y_train_0_2_activity_sub2_50_50.npy', y_train)
print("======loading segmented train_0_6 data=====")
#x_train = load('save_segmented_data/x_train_0_1_activity_huynh_5_5.npy')
#y_train = load('save_segmented_data/y_train_0_1_activity_huynh_5_5.npy')
y_train = le.fit_transform(y_train)
y_train = to_categorical(y_train, num_classes)
print('y_train shape: ', y_train.shape)
#labels_train = le.inverse_transform(np.argmax(y_train, axis=1))
#_, class_weighted_train, ratio_class_train = count_labels(labels_train)
# Set input & output dimensions
num_time_periods, num_sensors = x_train.shape[1], x_train.shape[2]
#print(list(le.classes_))
# Set input_shape / reshape for Keras
input_shape = (num_time_periods*num_sensors)
x_train = x_train.reshape(x_train.shape[0], input_shape)
print('x_train shape:', x_train.shape)
print('input_shape:', input_shape)
x_train = x_train.astype('float32')

test = read_data(cwd, args.test_file)
x_test, y_test, class_weight_test, ratio_class_test = create_segments_and_labels(test, TIME_PERIODS, STEP_DISTANCE)
#print("=======saving test_6 data==============")
#save('save_segmented_data/x_test_2_activity_sub2_50_50.npy', x_test)
#save('save_segmented_data/y_test_2_activity_sub2_50_50.npy', y_test)
print("=========loading segmented test_6 data===========")
#x_test = load('save_segmented_data/x_test_1_activity_huynh_5_5.npy')
#y_test = load('save_segmented_data/y_test_1_activity_huynh_5_5.npy')
x_test = x_test.reshape(x_test.shape[0], input_shape)
x_test = x_test.astype('float32')
y_test = le.fit_transform(y_test)
y_test = to_categorical(y_test, num_classes)
#labels_test = le.inverse_transform(np.argmax(y_test, axis=1))
#_, class_weighted_test, ratio_class_test = count_labels(labels_test)


model_m = keras.Sequential()
# Remark: since coreml cannot accept vector shapes of complex shape like
# [80,3] this workaround is used in order to reshape the vector internally
# prior feeding it into the network
model_m.add(layers.Reshape((TIME_PERIODS, N_FEATURES), input_shape=(input_shape,)))
model_m.add(layers.LSTM(100, activation='tanh', input_shape=(TIME_PERIODS, N_FEATURES)))
model_m.add(layers.Dropout(0.5))
model_m.add(layers.Dense(num_classes, activation='softmax'))
#print(model_m.summary())
#model_m.add(Reshape((TIME_PERIODS, 12), input_shape=(input_shape,)))
#model_m.add(LSTM(100, activation='tanh', input_shape=(TIME_PERIODS, 12)))
#model_m.add(Dropout(0.5))
#model_m.add(Dense(num_classes, activation='softmax'))

print(model_m.summary())

optimizer = optimizers.Adam(clipnorm=1)
model_m.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy'])


# Hyper-parameters
report_dir = "D:/Documents/Downloads/Human_Activity_Recognition/reports/S15"
os.makedirs(report_dir, exist_ok=True)
WEIGHTS_PATH = "D:/Documents/Downloads/Human_Activity_Recognition/weights/dalia-activity-wrist-acc-lstm/"
os.makedirs(WEIGHTS_PATH, exist_ok=True)
BATCH_SIZE = 64
EPOCHS = range(1, 16)
os.makedirs(report_dir, exist_ok=True)
for epoch_count in EPOCHS:
    print(f"Training for {epoch_count} epochs")

    # serialize model to JSON
    filepath = os.path.join(WEIGHTS_PATH, f"weights" + str(TIME_PERIODS) + str(STEP_DISTANCE) + "_activity.best.keras")
    #filepath="weights"+str(TIME_PERIODS)+str(STEP_DISTANCE)+"_activity.best.keras"
    checkpoint = callbacks.ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    # Enable validation to use ModelCheckpoint and EarlyStopping callbacks.
    history = model_m.fit(x_train,
                      y_train,
                      batch_size=BATCH_SIZE,
                      #validation_split=0.2,
                      epochs=epoch_count,
                      #class_weight = class_weight_train,
                      callbacks=callbacks_list,
                      verbose=1)


    #numpy.set_printoptions(threshold=sys.maxsize)

    #======Print confusion matrix for training data========
    y_pred_train = model_m.predict(x_train)
    #print('y_pred_train: ', y_pred_train)
    #Take the class with the highest probability from the train_0_6 predictions
    max_y_pred_train = np.argmax(y_pred_train, axis=1)
    max_train = np.argmax(y_train, axis=1)
    print(classification_report(max_train, max_y_pred_train))

    #write predicted activity list to file
    #act_list_predict_train = le.inverse_transform(max_y_pred_train)
    #write_list_to_file(act_list_predict_train,'semantic_concept_generation/activity_prediction_train.csv')

    #write actual activity list to file
    #act_list_actual_train = le.inverse_transform(max_train)
    #write_list_to_file(act_list_actual_train,'semantic_concept_generation/activity_actual_train.csv')


    # later...
    #=====load weights into s2.txt model====
    model_m.load_weights(filepath)
    print("Loaded model from disk")
    model_m.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    score = model_m.evaluate(x_test, y_test, verbose=1)

    print('\nAccuracy on test data: %0.2f' % score[1])
    print('\nLoss on test data: %0.2f' % score[0])

    y_pred_test = model_m.predict(x_test)
    #print('y_pred_test:', y_pred_test)
    #======Take the class with the highest probability from the test_6 predictions=====
    max_y_test = np.argmax(y_test, axis=1)

    max_y_pred_test = np.argmax(y_pred_test, axis=1)
    #print('F1 score:', classification.f1_score(max_y_test, max_y_pred_test, average='macro'))
    report = classification_report(max_y_test, max_y_pred_test, digits=2, output_dict=False)
    print(report)
    #show_confusion_matrix(max_y_test, max_y_pred_test)
    #print(classification_report(max_y_test, max_y_pred_test))
    report_file_path = os.path.join(report_dir,f"{report_dir}/LSTM_Wrist_batch_size{BATCH_SIZE}.txt")
    with open(report_file_path, "a") as text_file:
        text_file.write(f"Classification report for {epoch_count} epochs:\n")
        text_file.write(report)
        text_file.write("\n")


#====write predicted activity list to file====
#act_list_predict_test = le.inverse_transform(max_y_pred_test)
#act_list_predict_test = le.inverse_transform(max_y_pred_calibration)
#write_list_to_file(act_list_predict_test, 'semantic_concept_generation/activity_prediction_test.csv')

#====write actual activity list to file======
#act_list_actual_test = le.inverse_transform(max_y_test)
#write_list_to_file(act_list_actual_test, 'semantic_concept_generation/activity_actual_test.csv')

#=======================================================================================================
end_time = time.time()
total_time_in_seconds = end_time - start_time
print("Completion time took %.2f seconds" % total_time_in_seconds)
