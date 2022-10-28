# CRYPTOBOARD
![Devise Logo](https://github.com/p98a/crypto_board/blob/master/CryptoBoard_image.jpeg)

CryptoBoard, is an application which provides users a way to collaborate securely without being concerned about their data integrity and privacy being compromised.

Features:
* It is a Docker and Python based web application.
* It provides users the ability to grant access to boards which they are authorized to and maintains the others boards encrypted.
* The Data in the boards is encrypted before it is being saved into the database and the plain text is displayed only to the owner and the users who have access to it.


## Steps to start running CryptoBoard
Please follow below steps to start the implementation of CryptoBoard Application.

- Install Python 3.0 or higher, Django, Docker in your local system.
- Clone the code from the github link into your local system.
```
https://github.com/p98a/crypto_board
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
