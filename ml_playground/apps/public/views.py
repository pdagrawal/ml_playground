import os
from django.shortcuts import render, redirect
from django.http import HttpRequest, HttpResponse
from django.contrib import messages

import pandas as pd
import pickle
# import numpy as np
# import seaborn as sb
# import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import tree
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import confusion_matrix
# from sklearn.model_selection import train_test_split


def index(request: HttpRequest) -> HttpResponse:
    return render(request, "index.html")


def about(request: HttpRequest) -> HttpResponse:
    return render(request, "about.html")

def upload_csv(request: HttpRequest) -> HttpResponse:
    # return redirect('public:index')
    field_names = []
    if "GET" == request.method:
        return render(request, "public/index.html", field_names)
    # if not GET, then proceed
    try:
        csv_file = request.FILES["csv_file"]
        if not csv_file.name.endswith('.csv'):
            messages.error(request, 'File is not CSV type')
            return render(request, "public/index.html")
        #if file is too large, return
        if csv_file.multiple_chunks():
            messages.error(request, "Uploaded file is too big (%.2f MB)." % (csv_file.size/(1000*1000),))
            return render(request, "public/index.html")

        file_data = csv_file.read().decode("utf-8")

        lines = file_data.split("\n")
        field_names = lines[0].split(',')
        file_name = 'tmp/' + csv_file.name
        with open(file_name, 'wb+') as destination:
            for chunk in csv_file.chunks():
                destination.write(chunk)
        request.session['uploaded_file_path'] = file_name

        # file_hash = hash(file_name)
        # request.session[file_hash] = file_name
        #loop over the lines and save them in db. If error , store as string and then display
        # for line in lines:
        #     fields = line.split(",")
        #     data_dict = {}
        #     data_dict["name"] = fields[0]
        #     data_dict["start_date_time"] = fields[1]
        #     data_dict["end_date_time"] = fields[2]
        #     data_dict["notes"] = fields[3]
        #     try:
        #         form = EventsForm(data_dict)
        #         if form.is_valid():
        #             form.save()
        #         else:
        #             logging.getLogger("error_logger").error(form.errors.as_json())
        #     except Exception as e:
        #         logging.getLogger("error_logger").error(repr(e))
        #         pass

    except Exception as e:
        # logging.getLogger("error_logger").error("Unable to upload file. "+repr(e))
        messages.error(request, "Unable to upload file. "+repr(e))

    return render(request, "field_selection.html", {'field_names': field_names})

def work_in_progress(request: HttpRequest) -> HttpResponse:
    messages.success(request, f"Filepath: {request.session['uploaded_file_path']}")
    return render(request, "work_in_progress.html")

def create_svm_model(training_samples, clf_labels, **svm_params):
    svm_model = SVC(**svm_params)
    svm_model.fit(training_samples, clf_labels)

    with open("tmp/svm.pkl", 'wb') as f:
        pickle.dump(svm_model, f)

def create_multiple_regression_model(training_samples, clf_labels):
    multiple_reg=LinearRegression()  
    multiple_reg.fit(training_samples, clf_labels)

    with open("tmp/multiple.pkl",'wb') as f:
        pickle.dump(multiple_reg,f)

def create_logistic_regression_model(training_samples, clf_labels, **logistic_params):
    logistic=LogisticRegression(**logistic_params)  
    logistic.fit(training_samples, clf_labels)

    with open("tmp/logistic.pkl",'wb') as f:
        pickle.dump(logistic,f)

def create_decisiontree_model(training_samples, clf_labels, **dt_params):
    dt_model = tree.DecisionTreeClassifier(**dt_params)
    dt_model = dt_model.fit(training_samples, clf_labels)

    with open("tmp/decision.pkl", 'wb') as f:
        pickle.dump(dt_model, f)

def transform_dataset(df, dump_encoder=False, clf_param=""):
    if dump_encoder and os.path.exists('tmp/model_encoder.pkl'):
        os.remove('tmp/model_encoder.pkl')
    for column_name in df.columns:
        print(column_name)
        if df[column_name].dtype == object:
            enc = preprocessing.LabelEncoder().fit(df[column_name])
            df[column_name] = enc.transform(df[column_name])

            # Save the label encoder for future predictions
            if dump_encoder and column_name == clf_param:
                with open('tmp/model_encoder.pkl', 'wb') as file:
                    pickle.dump(enc, file, pickle.HIGHEST_PROTOCOL)
    return df


def process_dataset(dataset_file, training_params=[], clf_params=""):
    df = pd.read_csv(dataset_file)

    # Cleaning the data
    df = df.dropna()

    # Transform non-numeric columns
    df = transform_dataset(df, True, clf_params)
    # Get classification labels.
    clf_labels = df[clf_params]
    print(clf_labels.size)

    # Drop the classification labels to form training parameters.
    df.drop(clf_params, axis=1, inplace=True)

    # Keep features which are selected by the user.
    for param in list(set(df.columns.values) - set(training_params)):
        df.drop(param, axis=1, inplace=True)

    return [df, clf_labels]


def test(model_file, sample):
    with open(model_file, 'rb') as f:
        model = pickle.load(f)

    #predicted_value = model.predict(sample)
    predicted_value = [int(model.predict(sample))]
    if os.path.exists('tmp/model_encoder.pkl'):
        with open('tmp/model_encoder.pkl', 'rb') as handle:
            encoder = pickle.load(handle)
        
        predicted_value =  encoder.inverse_transform(predicted_value)

    
    return predicted_value

def train_model(request: HttpRequest) -> HttpResponse:
    attributes = request.POST.getlist('attributes')
    classification = request.POST.get("classification")
    algorithm  = request.POST.get("algorithm")
    # data = process_dataset("tmp/iris.csv", ["sepal_length", "sepal_width", "petal_length", "petal_width"], "class")
    print("========================")
    print(request.session['uploaded_file_path'])
    print("========================")
    data = process_dataset(request.session['uploaded_file_path'], attributes, classification)

    x_test = [[7.7, 2.6, 6.9, 2.3]]
    sample = pd.DataFrame(
        x_test, columns=attributes)
    sample = transform_dataset(sample)

    if algorithm == "SVM":

            svm_params = {'kernel': 'rbf'}
            create_svm_model(data[0], data[1], **svm_params)
            predicted_value = test("tmp/svm.pkl", sample)

    elif algorithm == "Decision Tree":
            dt_params = {}
            model = create_decisiontree_model(data[0], data[1], **dt_params)
            predicted_value = test("tmp/decision.pkl", sample)

    elif algorithm == "Multiple Regression":
            create_multiple_regression_model(data[0], data[1])
            predicted_value = test("tmp/multiple.pkl", sample)


    elif algorithm == "Logistic Regression":
            logistic_params = {'solver' : 'liblinear', 'random_state' : 0}
            model = create_logistic_regression_model(data[0], data[1], **logistic_params)
            predicted_value = test("tmp/logistic.pkl", sample)
        
    # x_test = [[1, 'Full-time', 'Mid-Senior level']]
    #x_test = [[7.7, 2.6, 6.9, 2.3]]
    #sample = pd.DataFrame(
    #    x_test, columns=attributes)
    #sample = transform_dataset(sample)
    #predicted_value = test("tmp/iris.pkl", sample)
    
    messages.success(request, "successfully predicted.")
    return render(request, "prediction_result.html", {'predicted_value': predicted_value})

    


def prediction_result(request: HttpRequest, predicted_value) -> HttpResponse:
    return render(request, "prediction_result.html", {'predicted_value': predicted_value})