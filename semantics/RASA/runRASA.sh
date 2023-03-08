#!/bin/sh

model_path=$(pwd)/trained
echo $model_path:/app

docker run -it -p 5005:5005 -v $model_path:/app \
  rasa/rasa:latest-full run --enable-api --port 5005
 
