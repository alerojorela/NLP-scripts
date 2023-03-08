#!/bin/sh

model_path=$(pwd)/trained
echo $model_path:/app

# TRAIN RASA
docker run \
  -v $model_path:/app \
  rasa/rasa:latest-full \
  train \
  --data data \
  --out models
  --domain domain.yml \
