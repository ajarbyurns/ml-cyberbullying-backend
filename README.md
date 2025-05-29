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
python app/ml/train/py
```

## API Endpoint(s)
```yaml
POST /predict #make predictions from text input
```

## Finish
In the terminal, press Ctrl-C and then enter the command below
```
docker-compose down
```