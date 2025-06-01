from app.ml.predict import Predictor

# Load latest model and encoder
predictor = Predictor()

# Test prediction
def test_predict(text):
    result = predictor.predict(text)

    print(f"Input: {text}")
    print(f"Predicted label: {result}")

if __name__ == "__main__":
    word = 'high school sucks'
    test_predict(word)