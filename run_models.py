from os.path import join as pjoin
import numpy as np
import pandas as pd
import pickle

import sys
print(sys.version)

import functions
from functions import rf, cnn

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

INPUT_DIR = pjoin("data", "ascii-sign-language")
SUBMISSION_TEMPLATE = pjoin("submissions", "predictions_{}.csv")
TRAINED_RF_PATH = pjoin("models", "trained_rf_depth10.pkl")

train = pd.read_csv(pjoin(INPUT_DIR, "sign_mnist_train.csv"))
valid = pd.read_csv(pjoin(INPUT_DIR, "sign_mnist_test.csv"))
test = pd.read_csv(pjoin(INPUT_DIR, "test.csv"), index_col=0)
print(train.shape, valid.shape, test.shape)

X_train, y_train = functions.split_off_labels(train)
X_valid, y_valid = functions.split_off_labels(valid)
test_index, test_data = functions.wrangle_test_set(test)
print(f"Test data shape after wrangling: {test_data.shape}")
adapt_args = {"test_sets": test_data, "index": test_index}


#%%
print(f"\nTraining CNN model")
cnn_model = cnn.ConvolutionalNeuralNet()
cnn_model.fit(
    X_train.reshape(X_train.shape[0], 28, 28, 1), 
    y_train,
    X_valid.reshape(X_valid.shape[0], 28, 28, 1), 
    y_valid,
    epochs = 1
)
cnn_preds = functions.adapt_preds_to_expectations(
    model = cnn_model,
    **{
        **adapt_args,
        "test_sets": test_data.reshape(2, test_data.shape[1], 28, 28, 1)
    }
)
cnn_preds.to_csv(SUBMISSION_TEMPLATE.format("cnn"))

#%%
print(f"\nLoading RF model")
from functions.rf import RandomForestClassifier, TreeClassifier
with open(TRAINED_RF_PATH, "rb") as f:
    rf_model = pickle.load(f)

rf_preds = functions.adapt_preds_to_expectations(
    model = rf_model,
    **adapt_args
)
rf_preds.to_csv(SUBMISSION_TEMPLATE.format("rf"))


#%%
print(f"\nTraining SVM model")
svm_model = SVC(random_state=5, kernel='rbf', C=1.0, gamma='scale')
svm_model.fit(X_train, y_train)
svm_preds = functions.adapt_preds_to_expectations(
    model = svm_model,
    **adapt_args
)
svm_preds.to_csv(SUBMISSION_TEMPLATE.format("svm"))