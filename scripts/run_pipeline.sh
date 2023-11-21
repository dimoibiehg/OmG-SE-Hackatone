#!/bin/bash

# Before running this file you should set the following permission: chmod +x /path/to/this_file.sh
# You can run this script from the command line using:
# ./run_pipeline.sh <start_date> <end_date> <raw_data_file> <processed_data_file> <model_file> <test_data_file> <predictions_file>
# For example:
# ./run_pipeline.sh 2020-01-01 2020-01-31 data/raw_data.csv data/processed_data.csv models/model.pkl data/test_data.csv predictions/predictions.json
# ./scripts/run_pipeline.sh "2022-01-01" "2023-01-01" "./data/fetched" "./data/processed_data.csv" "./models/model" "./data/processed_data.csv" "./predictions/predictions.json"


# Get command line arguments
start_date="$1"
end_date="$2"
raw_data_path="$3"
processed_data_file="$4"
model_file="$5"
test_data_file="$6"
predictions_file="$7"

# #Run data_ingestion.py
echo "Starting data ingestion..."
python3 src/data_ingestion.py --start_time="$start_date" --end_time="$end_date" --output_path="$raw_data_path"

# #Run data_processing.py
# echo "Starting data processing..."
# python3 src/data_processing.py --input_path="$raw_data_path" --output_file="$processed_data_file"

# # Run model_training.py
# echo "Starting model training..."
# python3 src/model_training.py --input_file="$processed_data_file" --model_file="$model_file"

# # # Run model_prediction.py
# echo "Starting prediction..."
# python3 src/model_prediction.py --input_file="$test_data_file" --model_file="$model_file" --output_file="$predictions_file"

echo "Pipeline completed."
