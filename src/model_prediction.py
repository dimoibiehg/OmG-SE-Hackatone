import pandas as pd
import argparse
import os
import re 
import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
import json

def load_data(file_path, ratio=0.2, window_size=200, store_file="./data/test.csv"):
    df: pd.DataFrame = pd.read_csv(file_path, index_col=False).iloc[:, 1:].astype(float)
    df = df.iloc[-int(ratio*df.shape[0])-window_size:, :] # based on the method, we need window_size of data extra than 20% of data
    df.to_csv(store_file)
    return df

def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

def make_predictions(df, model: tf.keras.Sequential, window_size=200):
    X_test = []
    y_test = []
    for i in range(df.shape[0]-window_size-1):
        X_test.append(df.iloc[i:(i+window_size), :-1].values.tolist())
        y_test.append(int(df.iloc[i+window_size, -1]))
    
    classes = model.predict(X_test)
    y_pred = [np.argmax(x) for x in classes]


    print(f"F1 Score (macro): {f1_score(y_test, y_pred, average='macro')}")
    print(f"F1 Score (micro): {f1_score(y_test, y_pred, average='micro')}")
    print(f"F1 Score (weighted): {f1_score(y_test, y_pred, average='weighted')}")

    return y_pred

def save_predictions(predictions, predictions_file="./predictions/predictions.json"):
    prediction_json = {"target": {f"{i}":int(x) for i,x in enumerate(predictions)}}
    with open(predictions_file, 'w') as fp:
        json.dump(prediction_json, fp, indent=2)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Prediction script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--input_file', 
        type=str, 
        default='data/test_data.csv', 
        help='Path to the test data file to make predictions'
    )
    parser.add_argument(
        '--model_file', 
        type=str, 
        default='models/model.pkl',
        help='Path to the trained model file'
    )
    parser.add_argument(
        '--output_file', 
        type=str, 
        default='predictions/predictions.json', 
        help='Path to save the predictions'
    )
    return parser.parse_args()

def main(input_file, model_file, output_file):
    df = load_data(input_file)
    
    model = load_model(model_file)
    predictions = make_predictions(df, model)
    save_predictions(predictions, output_file)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.model_file, args.output_file)
