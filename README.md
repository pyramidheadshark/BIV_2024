[![Static Badge](https://img.shields.io/badge/model_on_hugginface-link-black?style=for-the-badge&logo=huggingface&logoColor=black&labelColor=%23FFD21E&color=azure&link=https://huggingface.co/pyramidheadshark/rubert-turbo-payments)](https://huggingface.co/pyramidheadshark/rubert-turbo-payments)
[![Static Badge](https://img.shields.io/badge/License-Apache_2.0-%238FBC8F?style=for-the-badge&logo=apache&link=https://www.apache.org/licenses/LICENSE-2.0)](https://www.apache.org/licenses/LICENSE-2.0)
# ENG

# rubert-turbo-payments
A fast BERT model for multi-class payment classification. The model is based on [sergeyzh/rubert-tiny-turbo](https://huggingface.co/sergeyzh/rubert-tiny-turbo) - it has similar context size (2048), embedding size (312), and speed.

This repository contains all the tools for creating a dataset, training, and validating this model.

Trained on the BIV Payments dataset, provided at BIV Hack

## Installation

Directly from the repository

```bash
  docker build -t rubert_turbo_payments .
```

or

Using DockerHub
*[(link to DockerHub)](https://hub.docker.com/repository/docker/pyramidheadshark/ruberta_turbo_payments/general)*

```bash
  docker pull pyramidheadshark/ruberta_turbo_payments
```

или

Using GoogleDrive
*[(link to GoogleDrive)](https://drive.google.com/drive/folders/18iFmekzYOeHS3Kw9TsWq0fyMfcaOmE3-?usp=sharing)*

```bash
  cd [директория с rubert_turbo_payments.tar]
  docker load -i rubert_turbo_payments.tar
```
    
## Deployment

To run the model, create a directory for the container and place the dataset in the data folder, then execute

```bash
  docker run --rm -v [полный путь к директории для контейнера]\data:/app/data rubert_turbo_payments
```

---
# RU

# rubert-turbo-payments
Быстрая модель BERT для мультиклассовой классификации платежей. Модель основана на [sergeyzh/rubert-tiny-turbo](https://huggingface.co/sergeyzh/rubert-tiny-turbo) - имеет аналогичные размеры контекста (2048), эмбеддинга (312) и быстродействие

В данном репозитории имеется весь инструментарий для составления датасета, обучения и валидации данной модели

Обучено на датасете BIV Payments, предоставленным на BIV Hack

---

## Установка

Прямо из репозитория

```bash
  docker build -t rubert_turbo_payments .
```

или

Пользуясь DockerHub
*[(ссылка на DockerHub)](https://hub.docker.com/repository/docker/pyramidheadshark/ruberta_turbo_payments/general)*

```bash
  docker pull pyramidheadshark/ruberta_turbo_payments
```

или

С помощью GoogleDrive
*[(ссылка на GoogleDrive)](https://drive.google.com/drive/folders/18iFmekzYOeHS3Kw9TsWq0fyMfcaOmE3-?usp=sharing)*

```bash
  cd [директория с rubert_turbo_payments.tar]
  docker load -i rubert_turbo_payments.tar
```
    
## Развёртывание

Чтобы запустить модель, создайте директорию для контейнера и в папку data поместите датасет, а затем введите

```bash
  docker run --rm -v [полный путь к директории для контейнера]\data:/app/data rubert_turbo_payments
```

