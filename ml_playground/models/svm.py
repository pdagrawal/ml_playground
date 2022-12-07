import array
from itertools import cycle
import os
import pandas as pd
import pickle
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.metrics import RocCurveDisplay, accuracy_score, auc, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons


def plot_graph(clf, X, y):
    
    svc_disp = RocCurveDisplay.from_estimator(clf, X, y)
    svc_disp.plot(ax=plt.gca(), alpha=0.8)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for multi-class data')
    plt.legend(loc="lower right")
    plt.show()

def create_svm_model(training_samples, clf_labels, **svm_params):
    svm_model = SVC(**svm_params)
    svm_model.fit(training_samples, clf_labels)

    if os.path.exists('tmp/iris.pkl'):
        os.remove('tmp/iris.pkl')
    with open("tmp/iris.pkl", 'wb') as f:
        pickle.dump(svm_model, f)
    
    return svm_model
    #plot_graph(svm_model, training_samples.to_numpy(), clf_labels)

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

    with open("tmp/iris.pkl", 'wb') as f:
        pickle.dump(dt_model, f)


def transform_dataset(df, dump_encoder=False, clf_param=""):
    if dump_encoder and os.path.exists('tmp/iris_encoder.pkl'):
        os.remove('tmp/iris_encoder.pkl')
    
    encs = {}
    for column_name in df.columns:
        if df[column_name].dtype == object:
            encs[column_name] = preprocessing.LabelEncoder().fit(df[column_name])
            df[column_name] = encs[column_name].transform(df[column_name].astype(str))
            # Save the label encoder for future predictions
    if dump_encoder:
        with open('tmp/iris_encoder.pkl', 'wb') as file:
            pickle.dump(encs, file, pickle.HIGHEST_PROTOCOL)
    return df

def transform_test_dataset(df, dump_encoder=False, clf_param=""):
    if os.path.exists('tmp/iris_encoder.pkl'):
        with open('tmp/iris_encoder.pkl', 'rb') as handle:
            encoder = pickle.load(handle)

    for column_name in df.columns:
        if df[column_name].dtype == object:
            df[column_name] = encoder[column_name].transform(df[column_name])
    return df


def process_dataset(dataset_file, training_params=[], clf_params=""):
    df = pd.read_csv(dataset_file)
    df.head()
    unique_values = df[clf_params].unique()
    
    # Transform non-numeric columns
    df = transform_dataset(df, True, clf_params)

    # Get classification labels.
    clf_labels = df[clf_params]

    # Drop the classification labels to form training parameters.
    df.drop(clf_params, axis=1, inplace=True)

    # Keep features which are selected by the user.
    for param in list(set(df.columns.values) - set(training_params)):
        df.drop(param, axis=1, inplace=True)

    return [df, clf_labels, unique_values]

def test(model_file, sample):
    with open(model_file, 'rb') as f:
        model = pickle.load(f)

    predicted_value = model.predict(sample)
    if os.path.exists('tmp/iris_encoder.pkl'):
        with open('tmp/iris_encoder.pkl', 'rb') as handle:
            encoder = pickle.load(handle)
        predicted_value =  encoder['class'].inverse_transform(predicted_value)
    
    return predicted_value

def decode_labels(df) :
    if os.path.exists('tmp/iris_encoder.pkl'):
        with open('tmp/iris_encoder.pkl', 'rb') as handle:
            encoder = pickle.load(handle)
        df = encoder['class'].inverse_transform(df)

    return df

'''
cols = ['pregnant','glucose','bp','triceps','insulin','bmi','pedigree','age']
data = process_dataset("tmp/pima-indians-diabetes.csv",
                       cols, "diabetes")

#svm_params = {'kernel': 'rbf', 'probability': True}
#model = create_svm_model(data[0], data[1], **svm_params)

x_train, x_test, y_train, y_test = train_test_split(data[0], data[1], test_size=0.10, random_state=42)


svm_params = {'kernel': 'rbf'}
model = create_svm_model(x_train, y_train, **svm_params)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test,y_pred)*100
print('Accuracty Score ', accuracy)
print(y_pred)
cm = confusion_matrix(y_test, y_pred, labels=data[1].unique())
ax= plt.subplot()
sb.heatmap(cm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
plt.show()

i = 1
with open("tmp/pima-indians-diabetes.csv") as file:
    while (line := file.readline().rstrip()):
        if i == 1:
            i = i+1
            continue
        x_test = [line.split(',')]
        x_test[0] = x_test[0][:len(x_test[0])-1]
        x_test[0] = list(map(float, x_test[0])) 
        sample = pd.DataFrame(x_test, columns=cols)
        sample = transform_dataset(sample)
        #print(model.predict(sample))
        print(x_test, " \t", test("tmp/iris.pkl", sample))
        i = i+1
        if i == 100:
            break
'''

'''
cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
data = process_dataset("tmp/iris.csv",
                       ["sepal_length", "sepal_width", "petal_length", "petal_width"], "class")

#svm_params = {'kernel': 'rbf', 'probability': True}
#model = create_svm_model(data[0], data[1], **svm_params)

x_train, x_test, y_train, y_test = train_test_split(data[0], data[1], test_size=0.10, random_state=42)


svm_params = {'kernel': 'rbf'}
model = create_svm_model(x_train, y_train, **svm_params)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test,y_pred)*100
print('Accuracty Score ', accuracy)

print(y_test)
cm = confusion_matrix(y_test, y_pred, labels=data[1].unique())
ax= plt.subplot()
sb.heatmap(cm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
#plt.show()

x_test = [[7.7, 2.6, 6.9, 2.3]]
#x_test = [[6.3, 2.3, 4.4, 1.3]]
sample = pd.DataFrame(
    x_test, columns=cols)
sample = transform_dataset(sample)
print(test("tmp/iris.pkl", sample))

i = 1
with open("tmp/iris.csv") as file:
    while (line := file.readline().rstrip()):
        if i == 1:
            i = i+1
            continue
        x_test = [line.split(',')]
        x_test[0] = x_test[0][:len(x_test[0])-1]
        x_test[0] = list(map(float, x_test[0])) 
        sample = pd.DataFrame(x_test, columns=cols)
        sample = transform_dataset(sample)
        #print(model.predict(sample))
        print(x_test, " \t", test("tmp/iris.pkl", sample))
        i = i+1
        if i == 1:
            break
'''


cols=["cap-shape","cap-surface","cap-color","bruises","odor","gill-attachment","gill-spacing","gill-size","gill-color","stalk-shape","stalk-root","stalk-surface-above-ring","stalk-surface-below-ring","stalk-color-above-ring","stalk-color-below-ring","veil-type","veil-color","ring-number","ring-type","spore-print-color","population","habitat"]

data = process_dataset("static/mushrooms.csv", cols, "class")

x_train, x_test, y_train, y_test = train_test_split(data[0], data[1], test_size=0.10, random_state=42)


svm_params = {'kernel': 'rbf'}
model = create_svm_model(data[0], data[1], **svm_params)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)*100
print('Accuracty Score ', accuracy)

print(data[1].unique())
print(data[2])
cm = confusion_matrix(decode_labels(y_test), decode_labels(y_pred), labels=data[2])
ax= plt.subplot()
sb.heatmap(cm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(data[2]);ax.yaxis.set_ticklabels(data[2][::-1])
plt.savefig('tmp/cm.png')
plt.show()



i = 1
test_data = []
with open("static/mushrooms.csv") as file:
    while (line := file.readline().rstrip()):
        if i == 1:
            i = i+1
            continue
        x_test = [line.split(',')]
        test_data.append(x_test[0][1:])
        i = i+1
        if i == 10:
            break

sample = pd.DataFrame(test_data, columns=cols)
sample = transform_test_dataset(sample)
print(sample, " \t", test("tmp/iris.pkl", sample))


#dt_params = {}
#model = create_decisiontree_model(data[0], data[1], **dt_params)
'''
x_test = [[7.7, 2.6, 6.9, 2.3]]
#x_test = [[6.3, 2.3, 4.4, 1.3]]
sample = pd.DataFrame(
    x_test, columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
sample = transform_dataset(sample)
'''
#test("tmp/iris_dt.pkl", sample)



#logistic_params = {'solver' : 'liblinear', 'random_state' : 0}
#model = create_logistic_regression(data[0], data[1], **logistic_params)
'''
x_test = [[7.7, 2.6, 6.9, 2.3]]
#x_test = [[6.3, 2.3, 4.4, 1.3]]
sample = pd.DataFrame(
    x_test, columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
sample = transform_dataset(sample)
'''
#test("tmp/logistic.pkl", sample)


#mr_model = create_multiple_regression(data[0], data[1])
'''
x_test = [[7.7, 2.6, 6.9, 2.3]]
#x_test = [[6.3, 2.3, 4.4, 1.3]]
sample = pd.DataFrame(
    x_test, columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
sample = transform_dataset(sample)
'''
#test("tmp/multiple.pkl", sample)