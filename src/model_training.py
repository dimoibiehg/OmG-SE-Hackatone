import pandas as pd
import argparse
from utils.model_training_utils import * 
"""
ratio: define ratio of the trainig data
"""
def load_data(file_path, ratio=0.8, store_file="./data/train.csv"):
    df: pd.DataFrame = pd.read_csv(file_path, index_col=False).iloc[:, 1:].astype(float)
    df = df.iloc[:int(ratio*df.shape[0]), :]
    df.to_csv(store_file)
    return df

"""
this function make a window shapes data from the list data
"""
def split_data(data: pd.DataFrame, window_size = 200):
    X_train = []
    y_train = []
    for i in range(data.shape[0]-window_size-1):
        X_train.append(data.iloc[i:(i+window_size), :-1].values.tolist())
        y_train.append(int(data.iloc[i+window_size, -1]))
    
    return X_train, y_train
    

def train_model(X_train, y_train):
    sqe_len = len(X_train[0])
    feature_dim = len(X_train[0][0])
    model = build_convolutional_model(sqe_len, feature_dim)
    model.fit(np.array(X_train),np.array(y_train),
                        epochs=1000, batch_size=32,validation_split=0.2,
                        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=30, mode="min")])
    return model

def save_model(model: keras.Sequential, model_path):
    model.save(model_path)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Model training script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--input_file', 
        type=str, 
        default='data/processed_data.csv', 
        help='Path to the processed data file to train the model'
    )
    parser.add_argument(
        '--model_file', 
        type=str, 
        default='models/model', 
        help='Path to save the trained model'
    )
    return parser.parse_args()

def main(input_file, model_file):
    df = load_data(input_file)
    X_train, y_train = split_data(df)
    print("start training")
    model: keras.Sequential = train_model(X_train, y_train)
    print(model.summary())
    save_model(model, model_file)


if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.model_file)