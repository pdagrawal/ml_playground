# ML PLAYGROUND

ML playground is a web-platform that allows users to interact with machine learning algorithms from the browser. Users can upload any dataset, select a model, configure features and interact with the model by providing the test data.

Features:
* Users can upload the dataset in csv, xlsx file format.
* Users will have flexibility to select different ML algorithms to train the model.
* Supports Support Vector Machine, Multiple Regression, Decision Tree and Logistic Regression ML algorithms.
* Users will have option to specify features to train the model.
* After training the model, user will be able to interact with the model by providing test samples.
  
Architecture:
![image](https://github.com/pdagrawal/ml_playground/assets/20897894/383cdb15-d54a-4356-97d5-3fa3a94f7084)

Interactive UI:

![image](https://github.com/pdagrawal/ml_playground/assets/20897894/ffbba95a-b250-4986-b23e-145576397d58)
![image](https://github.com/pdagrawal/ml_playground/assets/20897894/b8e07808-44ee-4124-b4c1-c1bd9f25f710)
![image](https://github.com/pdagrawal/ml_playground/assets/20897894/08cd9b5c-e220-4afb-937a-e6eecf6b8480)

## Steps to start running ML Playground
Please follow below steps to start the implementation of ml_playground Application.

- Install Python 3.0 or higher, Django, Docker in your local system.
- Clone the code from the github link into your local system.
```
https://github.com/pdagrawal/ml_playground
```
- Open Docker and execute the 2 commands given below to setup the application environment.
Command 1:
```
make build
```
The docker build command builds Docker images from a Dockerfile and a “context”. A build's context is the set of files located in the specified PATH or URL
Command 2:
```
make compose start
```
The compose up command aggregates the output of each container (essentially running docker-compose logs --follow). When the command exits, all containers are stopped. Running docker-compose up --detach starts the containers in the background and leaves them running.
- Now go to the url given below for accessing the Home page of the application.
```
http://localhost:8000/
```


### NOTE:

For Windows operarting systems, please install GUI compiler and enbale hyper-v in your system to run the application.
