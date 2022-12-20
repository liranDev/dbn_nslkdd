import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from dbn.tensorflow import SupervisedDBNClassification

y_df = pd.read_csv('../dbn-based-nids/data/processed_nslkdd/train/train_labels.csv').iloc[:, 1:]

x_df = pd.read_csv('../dbn-based-nids/data/processed_nslkdd/train/train_features.csv').iloc[:, 1:]

X_test = pd.read_csv('../dbn-based-nids/data/processed_nslkdd/test/test_features.csv').iloc[:, 1:]
Y_test = pd.read_csv('../dbn-based-nids/data/processed_nslkdd/test/test_labels.csv').iloc[:, 1:]

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
