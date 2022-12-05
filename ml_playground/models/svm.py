import pandas as pd
import pickle
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


def create_svm_model(training_samples, clf_labels, **svm_params):
    svm_model = SVC(**svm_params)
    svm_model.fit(training_samples, clf_labels)

    with open("tmp/iris.pkl", 'wb') as f:
        pickle.dump(svm_model, f)

def create_multiple_regression(training_samples, clf_labels):
    lin_reg=LinearRegression()  
    lin_reg.fit(training_samples, clf_labels)

    with open("tmp/multiple.pkl",'wb') as f:
        pickle.dump(lin_reg,f)

def create_logistic_regression(training_samples, clf_labels, **logistic_params):
    logistic=LogisticRegression(**logistic_params)  
    logistic.fit(training_samples, clf_labels)

    with open("tmp/logistic.pkl",'wb') as f:
        pickle.dump(logistic,f)

def create_decisiontree_model(training_samples, clf_labels, **dt_params):
    dt_model = tree.DecisionTreeClassifier(**dt_params)
    dt_model = dt_model.fit(training_samples, clf_labels)

    with open("tmp/iris_dt.pkl", 'wb') as f:
        pickle.dump(dt_model, f)


def transform_dataset(df, dump_encoder=False, clf_param=""):
    for column_name in df.columns:
        if df[column_name].dtype == object:
            enc = preprocessing.LabelEncoder().fit(df[column_name])
            df[column_name] = enc.transform(df[column_name])
            
            # Save the label encoder for future predictions
            if dump_encoder and column_name == clf_param:
                with open('tmp/iris_encoder.pkl', 'wb') as file:
                    pickle.dump(enc, file, pickle.HIGHEST_PROTOCOL)
    return df


def process_dataset(dataset_file, training_params=[], clf_params=""):
    df = pd.read_csv(dataset_file)
    df.head()

    # Transform non-numeric columns
    df = transform_dataset(df, True, clf_params)

    # Get classification labels.
    clf_labels = df[clf_params]

    # Drop the classification labels to form training parameters.
    df.drop(clf_params, axis=1, inplace=True)

    # Keep features which are selected by the user.
    for param in list(set(df.columns.values) - set(training_params)):
        df.drop(param, axis=1, inplace=True)

    return [df, clf_labels]


def test(model_file, sample):
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    
    with open('tmp/iris_encoder.pkl', 'rb') as handle:
        encoder = pickle.load(handle)

    print(encoder.inverse_transform([int(model.predict(sample))]))



data = process_dataset("tmp/iris.csv",
                       ["sepal_length", "sepal_width", "petal_length", "petal_width"], "class")

svm_params = {'kernel': 'rbf'}
model = create_svm_model(data[0], data[1], **svm_params)

x_test = [[7.7, 2.6, 6.9, 2.3]]
#x_test = [[6.3, 2.3, 4.4, 1.3]]
sample = pd.DataFrame(
    x_test, columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
sample = transform_dataset(sample)
test("tmp/iris.pkl", sample)


dt_params = {}
model = create_decisiontree_model(data[0], data[1], **dt_params)
'''
x_test = [[7.7, 2.6, 6.9, 2.3]]
#x_test = [[6.3, 2.3, 4.4, 1.3]]
sample = pd.DataFrame(
    x_test, columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
sample = transform_dataset(sample)
'''
test("tmp/iris_dt.pkl", sample)



logistic_params = {'solver' : 'liblinear', 'random_state' : 0}
model = create_logistic_regression(data[0], data[1], **logistic_params)
'''
x_test = [[7.7, 2.6, 6.9, 2.3]]
#x_test = [[6.3, 2.3, 4.4, 1.3]]
sample = pd.DataFrame(
    x_test, columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
sample = transform_dataset(sample)
'''
test("tmp/logistic.pkl", sample)


mr_model = create_multiple_regression(data[0], data[1])
'''
x_test = [[7.7, 2.6, 6.9, 2.3]]
#x_test = [[6.3, 2.3, 4.4, 1.3]]
sample = pd.DataFrame(
    x_test, columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
sample = transform_dataset(sample)
'''
test("tmp/multiple.pkl", sample)