import tensorflow as tf
import argparse
import numpy as np

def init_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, required=True)
    return parser

def load_model(path: str):
    try:
        model = tf.keras.models.load_model(path)
        return model
    except Exception as e:
        print(f"Failed to load model: {e}")

def read_input() -> str:
    print("\nEnter text:")
    user_input = input("> ")
    return user_input.strip()

def analyze_sentiment(model, text: str) -> str:
    input_data = tf.constant([text])
    predictions = model.predict(input_data, verbose=0)
    score = predictions[0][0]
    
    prob = tf.sigmoid(score).numpy()

    if score >= 0.5:
        return f"Positive ({(prob * 100):.2f}%)"
    else:
        return f"Negative ({((1 - prob) * 100):.2f}%)"

def main():
    parser = init_args()
    args = parser.parse_args()
    
    model = load_model(args.path)
    
    user_input = read_input()
    print(f'Result: \n{analyze_sentiment(model, user_input)}')

if __name__ == '__main__':
    main()