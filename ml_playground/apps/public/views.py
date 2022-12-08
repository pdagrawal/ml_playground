import os
from django.shortcuts import render, redirect
from django.http import HttpRequest, HttpResponse
from django.contrib import messages
import logging

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

logger = logging.getLogger(__name__)

def index(request: HttpRequest) -> HttpResponse:
    request.session['uploaded_file_path'] = None
    request.session['joined_parameters'] = None
    request.session['selected_algorithm'] = None
    request.session['predicted_value'] = None
    return render(request, "index.html")

def about(request: HttpRequest) -> HttpResponse:
    return render(request, "about.html")

def upload_dataset(request: HttpRequest) -> HttpResponse:
    # return redirect('public:index')
    field_names = []
    if "GET" == request.method:
        return render(request, "public/index.html", field_names)
    # if not GET, then proceed
    try:
        dataset_file = request.FILES["dataset_file"]
        #if file is too large, return
        if dataset_file.multiple_chunks():
            messages.error(request, "Uploaded file is too big (%.2f MB)." % (dataset_file.size/(1000*1000),))
            return render(request, "public/index.html")

        if dataset_file.name.endswith('.csv'):
            file_data = pd.read_csv(dataset_file)
        elif dataset_file.name.endswith('.xlsx'):
            file_data = pd.read_excel(dataset_file)
        else:
            messages.error(request, 'Only CSV and xlsx filetypes supported!')
            return render(request, "public/index.html")

        column_names = file_data.head()
        file_name = 'tmp/' + dataset_file.name
        with open(file_name, 'wb+') as destination:
            for chunk in dataset_file.chunks():
                destination.write(chunk)
        request.session['uploaded_file_path'] = file_name
    except Exception as e:
        messages.error(request, "Unable to upload file. "+repr(e))

    return render(request, "field_selection.html", {'column_names': column_names})

def create_svm_model(training_samples, clf_labels, **svm_params):
    svm_model = SVC(**svm_params)
    svm_model.fit(training_samples, clf_labels)

    with open("tmp/svm.pkl", 'wb') as f:
        pickle.dump(svm_model, f)

    return svm_model

def create_multiple_regression_model(training_samples, clf_labels):
    multiple_reg = LinearRegression()
    multiple_reg.fit(training_samples, clf_labels)

    with open("tmp/multiple_regression.pkl",'wb') as f:
        pickle.dump(multiple_reg,f)

    return multiple_reg

def create_logistic_regression_model(training_samples, clf_labels, **logistic_params):
    logistic = LogisticRegression(**logistic_params)
    logistic.fit(training_samples, clf_labels)

    with open("tmp/logistic_regression.pkl",'wb') as f:
        pickle.dump(logistic,f)

    return logistic

def create_decisiontree_model(training_samples, clf_labels, **dt_params):
    dt_model = tree.DecisionTreeClassifier(**dt_params)
    dt_model = dt_model.fit(training_samples, clf_labels)

    with open("tmp/decision_tree.pkl", 'wb') as f:
        pickle.dump(dt_model, f)

    return dt_model

def transform_dataset(df, dump_encoder=False, clf_param=""):
    if dump_encoder and os.path.exists('tmp/model_encoder.pkl'):
        os.remove('tmp/model_encoder.pkl')

    encs = {}
    for column_name in df.columns:
        if df[column_name].dtype == object:
            encs[column_name] = preprocessing.LabelEncoder().fit(df[column_name])
            df[column_name] = encs[column_name].transform(df[column_name].astype(str))
            # Save the label encoder for future predictions
    if dump_encoder:
        with open('tmp/model_encoder.pkl', 'wb') as file:
            pickle.dump(encs, file, pickle.HIGHEST_PROTOCOL)
    return df

def transform_test_dataset(df, dump_encoder=False, clf_param=""):
    if os.path.exists('tmp/model_encoder.pkl'):
        with open('tmp/model_encoder.pkl', 'rb') as handle:
            encoder = pickle.load(handle)

    for column_name in df.columns:
        if df[column_name].dtype == object:
            df[column_name] = encoder[column_name].transform(df[column_name])
    return df


def process_dataset(request, dataset_file, training_params=[], classification=""):
    if dataset_file.endswith('.csv'):
        df = pd.read_csv(dataset_file)
    elif dataset_file.endswith('.xlsx'):
        df = pd.read_excel(dataset_file)

    df = df.dropna()
    unique_values = df[classification].unique()
    logger.info(df.describe())
    logger.info(df.info())

    logger.info("===========train_model: Dtypes=============")
    logger.info(df.dtypes)
    logger.info("========================")

    dtypes = []
    for k, v in dict(df.dtypes).items():
        if k in training_params:
            dtypes.append(str(v))

    request.session['selected_parameters_dtypes'] = ','.join(dtypes)

    # Transform non-numeric columns
    df = transform_dataset(df, True, classification)
    # Get classification labels.
    clf_labels = df[classification]

    # Drop the classification labels to form training parameters.
    df.drop(classification, axis=1, inplace=True)

    # Keep features which are selected by the user.
    for param in list(set(df.columns.values) - set(training_params)):
        df.drop(param, axis=1, inplace=True)

    return [df, clf_labels, unique_values]


def test(model_file, sample, classification: str):
    with open(model_file, 'rb') as f:
        model = pickle.load(f)

    predicted_value = [int(model.predict(sample))]
    if os.path.exists('tmp/model_encoder.pkl'):
        with open('tmp/model_encoder.pkl', 'rb') as handle:
            encoder = pickle.load(handle)
        if classification in encoder:
            predicted_value = encoder[classification].inverse_transform(predicted_value)

    return predicted_value

def decode_labels(df, classification: str) :
    if os.path.exists('tmp/model_encoder.pkl'):
        with open('tmp/model_encoder.pkl', 'rb') as handle:
            encoder = pickle.load(handle)
        if classification in encoder:
            df = encoder[classification].inverse_transform(df)

    return df

def create_cm(df, model, classification):
    _x_train, x_test, _y_train, y_test = train_test_split(df[0], df[1], test_size=0.20, random_state=42)

    y_pred = model.predict(x_test)
    logger.info(y_pred)
    logger.info(y_test)
    accuracy = accuracy_score(y_test, y_pred)*100
    logger.info(f'Accuracty Score: {accuracy}')

    cm = confusion_matrix(decode_labels(y_test, classification), decode_labels(y_pred, classification), labels=df[2])
    ax = plt.subplot()
    sb.heatmap(cm, annot=True, fmt='g', ax=ax)

    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(df[2])
    ax.yaxis.set_ticklabels(df[2])
    plt.savefig('static/img/confusion_matrix.png')
    return round(accuracy, 2)

def train_model(request: HttpRequest) -> HttpResponse:
    logger.info(request.POST)
    attributes = request.POST.getlist('attributes')
    joined_parameters = ','.join(attributes)
    request.session['joined_parameters'] = joined_parameters
    logger.info(joined_parameters)
    classification = request.POST.get("classification")
    algorithm  = request.POST.get("algorithm")
    data = process_dataset(request, request.session['uploaded_file_path'], attributes, classification)

    if algorithm == "svm":
        svm_params = {'kernel': 'rbf'}
        model = create_svm_model(data[0], data[1], **svm_params)
    elif algorithm == "decision_tree":
        dt_params = {}
        model = create_decisiontree_model(data[0], data[1], **dt_params)
    elif algorithm == "multiple_regression":
        model = create_multiple_regression_model(data[0], data[1])
    elif algorithm == "logistic_regression":
        logistic_params = {'solver' : 'liblinear', 'random_state' : 0}
        model = create_logistic_regression_model(data[0], data[1], **logistic_params)

    accuracy = create_cm(data, model, classification)
    request.session['cm_accuracy_score'] = accuracy

    request.session['selected_algorithm'] = algorithm
    request.session['selected_classification'] = classification
    messages.success(request, "Model trained succcessfully.")
    return redirect('public:test_model')

def test_model(request):
    if request.method == 'GET':
        joined_parameters = request.session['joined_parameters']
        column_names = joined_parameters.split(',')
        return render(request, "test_model.html", {'column_names': column_names, 'accuracy': request.session['cm_accuracy_score']})
    elif request.method == 'POST':
        logger.info(request.POST)
        data = dict(request.POST.items())
        del data['csrfmiddlewaretoken']

        # x_test = [[7.7, 2.6, 6.9, 2.3]]  # Iris-virginica
        # x_test = [['Mid-Senior level', "Bachelor's Degree", "Financial Services"]]  # 0
        # x_test = [['Mid-Senior level', "High School or equivalent", "Oil & Energy"]]  # 1
        # x_test = [['x','s','y','t','a','f','c','b','k','e','c','s','s','w','w','p','w','o','p','n','n','g']]  # e
        # x_test = [['x','y','w','t','p','f','c','n','n','e','e','s','s','w','w','p','w','o','p','k','s','u']]  # p

        logger.info("===========Data=============")
        logger.info(data)
        logger.info("========================")

        joined_parameters = request.session['joined_parameters']
        column_names = joined_parameters.split(',')
        dtypes = request.session['selected_parameters_dtypes'].split(',')
        logger.info("===========Dtypes=============")
        logger.info(dtypes)
        logger.info("========================")
        for i in range(len(dtypes)):
            if dtypes[i] == 'float64':
                data[column_names[i]] = float(data[column_names[i]])
            elif dtypes[i] == 'int64':
                data[column_names[i]] = int(data[column_names[i]])
            else:
                data[column_names[i]] = str(data[column_names[i]])

        x_test = [data.values()]
        sample = pd.DataFrame(x_test, columns=column_names)
        sample = transform_test_dataset(sample)
        predicted_value = test(f"tmp/{request.session['selected_algorithm']}.pkl", sample, request.session['selected_classification'])
        request.session['predicted_value'] = str(predicted_value)
        messages.success(request, "Classification successful.")
        return redirect('public:prediction_result')


def prediction_result(request: HttpRequest) -> HttpResponse:
    return render(request, "prediction_result.html", {'predicted_value': request.session['predicted_value']})
