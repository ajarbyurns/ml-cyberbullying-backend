# NLP Text Classification API
The task is to take a text/String as an input and evaluate which of the following categories the text/String belongs to:

- Not Cyberbullying 
- Gender
- Ethnicity
- Religion
- Other types of Cyberbullying

Credits to https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification/data for the training dataset.

## Run with Docker

```
docker-compose up --build
```
## Model Training

```
python3 app/ml/train.py
```

## API Endpoint(s)
```
curl -X POST http://127.0.0.1:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "I hate you!"}'
```

## Finish (if you use Docker)
In the terminal, press Ctrl-C and then enter the command below
```
docker-compose down
```
## Sample
![](/Users/barryjuans/Downloads/Screenshot 2025-06-01 at 20.57.32.png)