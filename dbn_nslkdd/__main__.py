import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import os
from dbn.tensorflow import SupervisedDBNClassification


DATA_DIR = '../data'
FILE_PATH_TRAIN_FEATURES = os.path.join(DATA_DIR, 'processed_nslkdd', 'train/train_features.csv')
FILE_PATH_TRAIN_LABLES = os.path.join(DATA_DIR, 'processed_nslkdd', 'train/train_labels.csv')

FILE_PATH_TEST_FEATURES = os.path.join(DATA_DIR, 'processed_nslkdd', 'test/test_features.csv')
FILE_PATH_TEST_LABLES = os.path.join(DATA_DIR, 'processed_nslkdd', 'test/test_labels.csv')

y_df = pd.read_csv(FILE_PATH_TRAIN_FEATURES).iloc[:, 1:]
x_df = pd.read_csv(FILE_PATH_TRAIN_LABLES).iloc[:, 1:]

X_test = pd.read_csv(FILE_PATH_TEST_FEATURES).iloc[:, 1:]
Y_test = pd.read_csv(FILE_PATH_TEST_LABLES).iloc[:, 1:]

X1 = np.array(x_df)
Y1 = np.array(y_df).transpose()[0]

n_epochs_rbm = 2
n_iter_backprop = 10

classifier = SupervisedDBNClassification(hidden_layers_structure=[256, 256],
                                         learning_rate_rbm=0.05,
                                         learning_rate=0.1,
                                         n_epochs_rbm=n_epochs_rbm,
                                         n_iter_backprop=n_iter_backprop,
                                         batch_size=32,
                                         activation_function='relu',
                                         dropout_p=0.2)

classifier.fit(X1, Y1)

Y_pred = classifier.predict(X_test)
print('Done.\nAccuracy: %f' % accuracy_score(Y_test, Y_pred))
