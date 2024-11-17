# rubert-turbo-payments

## Сcылки на Docker Image
# *[(ссылка на DockerHub)](https://hub.docker.com/repository/docker/pyramidheadshark/ruberta_turbo_payments/general)*

или
# *[(ссылка на GoogleDrive)](https://drive.google.com/drive/folders/18iFmekzYOeHS3Kw9TsWq0fyMfcaOmE3-?usp=sharing)*

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




