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

    with open("tmp/iris.pkl", 'wb') as f:
        pickle.dump(svm_model, f)


def transform_dataset(df):
    le = preprocessing.LabelEncoder()
    for column_name in df.columns:
        if df[column_name].dtype == object:
            df[column_name] = le.fit_transform(df[column_name])
    return df


def process_dataset(dataset_file, training_params=[], clf_params=""):
    df = pd.read_csv(dataset_file)
    df.head()

    # Transform non-numeric columns
    df = transform_dataset(df)
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
    return model.predict(sample)

def train_model(request: HttpRequest) -> HttpResponse:
    print('=====================')
    print(dict(request.POST.items()))
    print('=====================')
    print(dict(request.POST.getlist('attributes')))
    request.POST.getlist('attributes')
    # print(request.POST.get("classification"))
    # print(request.POST.get("algorithm"))
    print('=====================')
    data = process_dataset("tmp/iris.csv", ["sepal_length", "sepal_width", "petal_length", "petal_width"], "class")
    # data = process_dataset(request.session['uploaded_file_path'], ["sepal_length", "sepal_width", "petal_length", "petal_width"], "class")

    svm_params = {'kernel': 'rbf'}
    create_svm_model(data[0], data[1], **svm_params)

    x_test = [[7.7, 2.6, 6.9, 2.3]]
    sample = pd.DataFrame(
        x_test, columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
    sample = transform_dataset(sample)
    predicted_value = test("tmp/iris.pkl", sample)
    messages.success(request, "successfully predicted.")
    return redirect('public:prediction_result', predicted_value = predicted_value)

def prediction_result(request: HttpRequest, predicted_value) -> HttpResponse:
    return render(request, "prediction_result.html", {'predicted_value': predicted_value})