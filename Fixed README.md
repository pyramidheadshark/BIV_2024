[![Static Badge](https://img.shields.io/badge/model_on_hugginface-link-black?style=for-the-badge&logo=huggingface&logoColor=black&labelColor=%23FFD21E&color=azure&link=https://huggingface.co/pyramidheadshark/rubert-turbo-payments)](https://huggingface.co/pyramidheadshark/rubert-turbo-payments)
[![Static Badge](https://img.shields.io/badge/License-Apache_2.0-%238FBC8F?style=for-the-badge&logo=apache&link=https://www.apache.org/licenses/LICENSE-2.0)](https://www.apache.org/licenses/LICENSE-2.0)
# rubert-turbo-payments

**Быстрая модель BERT для мультиклассовой классификации платежей**

Модель основана на [sergeyzh/rubert-tiny-turbo](https://huggingface.co/sergeyzh/rubert-tiny-turbo) - имеет аналогичные размеры контекста *(2048)*, эмбеддинга *(312)* и быстродействия при запуске на CPU


## Установка

Прямо из репозитория

```bash
  docker build -t rubert_turbo_payments .
```
или
Используя готовый Image
```bash
  cd [директория с rubert_turbo_payments.tar]
  docker load -i rubert_turbo_payments.tar
```
    
## Развёртывание

Чтобы запустить модель, введите

```bash
  docker run --rm -v [полный путь к директории для контейнера]:/app/data rubert_turbo_payments
```



