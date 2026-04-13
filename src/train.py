from sklearn.model_selection import train_test_split
from src.model import build_model

def train(X, y):
    print("Splitting data...")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print("Building model...")
    model = build_model()

    print("Training model...")
    model.fit(X_train, y_train, epochs=3, verbose=1)

    loss, acc = model.evaluate(X_test, y_test)
    print("Accuracy:", acc)

    model.save("models/model.keras")