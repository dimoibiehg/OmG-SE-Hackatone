import argparse
import pandas as pd
from utils.data_processing_utils import * 
from monitor import DATA_TRACER, SPAN_PROCESSOR, OLTP_EXPORTER
import os
import re
import warnings
import pickle
from utils.common import *

def load_data(file_path):
    df = pd.read_csv(file_path, header=0)
    return df

"""
target_cols: indicates columns that should be considered for filling missing data,
             the last one should be the energy quantity 
"""
def clean_data(df, target_cols, 
               file_type, properties: dict,
               start_time=datetime.datetime.strptime("2022-01-01T00:00+00:00Z", '%Y-%m-%dT%H:%M%zZ'),
               end_time=datetime.datetime.strptime("2023-01-01T00:00+00:00Z", '%Y-%m-%dT%H:%M%zZ')):
    
    df_clean = unify_column_names(df)

    # count Nan values and add it as a telemetry
    val = int(df_clean[target_cols].isna().sum().sum())
    if(val > 0):
        with DATA_TRACER.start_as_current_span(__name__) as span:
            properties["value"] = val
            span.add_event(f"missing_value_in_{file_type}", properties)

    df_clean = make_correct_data_types(df_clean)
    df_clean = remove_out_of_time_limit_data(df_clean, start_time, end_time)
    df_clean[target_cols[-1]] = filling_missing_data(df_clean[target_cols[-1]])
    df_clean = make_energy_units_unified(df_clean, qunatity_column_name=target_cols[-1])
    return df_clean

def preprocess_data(df, 
                    start_time=datetime.datetime.strptime("2022-01-01T00:00+00:00Z", '%Y-%m-%dT%H:%M%zZ'),
                    end_time=datetime.datetime.strptime("2023-01-01T00:00+00:00Z", '%Y-%m-%dT%H:%M%zZ')):
    new_data = group_data_points(df)
    new_data = fill_zero_not_exist_timesteps(new_data, start_time, end_time)
    return  new_data


def save_data(df: pd.DataFrame, output_file):
    df.to_csv(output_file)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Data processing script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--input_path',
        type=str,
        default='./data',
        help='Path to the raw data file to process'
    )
    parser.add_argument(
        '--output_file', 
        type=str, 
        default='data/processed_data.csv', 
        help='Path to save the processed data'
    )
    return parser.parse_args()

"""
patterns: list of patterns for files which have been fetched for green energy and load of countries, resppectively.
          So, the first pattern should detect country code and its corresponding block. 
          The second code should detect the country code.
"""
def main(input_path, output_file, 
         target_cols = ["StartTime", "EndTime", "quantity"], 
         block_filter = ["B01", "B09", "B10", "B11", "B12", "B13", "B15", "B16", "B18", "B19"],
         patterns=[r"gen_(\w{2})_([A-Za-z0-9]+).csv", r"load_(\w{2}).csv"]):
    
    files_addrs = os.listdir(input_path) # detecting all fetched files in the output_path
    total_df: pd.DataFrame = None # whole valid data for training and testing
    # applying patterns for detecting type of each file
    for  addr in files_addrs: 
        for i, pattern in enumerate(patterns):
            match = re.match(pattern, addr)
            if match:
                
                #first stage in data processing
                df = load_data(f"{input_path}/{addr}")
                
                # extracting telemetry info for the rest of the process
                file_name = ""
                country_code = str(match.group(1))
                properties = {"country_cdoe": country_code}
                num_of_detected_args = len(match.groups())
                if(num_of_detected_args == 2):
                    file_type = "green_energy"
                    block_code = str(match.group(2))
                    properties["block_code"] =  block_code
                    # based on the hint of the hackatone, just using blocks in the dataset
                    if(block_code not in block_filter):
                        continue
                elif( num_of_detected_args == 1):
                    file_type = "load"
                else:
                    warnings.warn(f"The following file is not identified by the specified patterns: {addr}")

                #second stage in data processing
                df_clean = clean_data(df, target_cols, file_type, properties)
                #third stage in data processing
                df_processed = preprocess_data(df_clean)
                print(df_processed.shape)
                # creating dataset for training and testing
                column_name = f"{file_type}_{country_code}"
                if(total_df is None):
                    total_df = df_processed.copy()
                    total_df.rename(columns={target_cols[-1]:column_name}, inplace=True)
                else:
                    if(column_name in list(total_df.columns)):
                        total_df[column_name] += df_processed[target_cols[-1]]
                    else:
                        total_df[column_name] = df_processed[target_cols[-1]]

                break
    

    # uncomment this line to replace the stored data
    # store_in_pickle("./data/total_df.pkl",total_df)

    # uncomment this line if you want skip the above process
    # total_df = retrieve_from_pickle("./data/total_df.pkl")

    country_dict  =  {
    "SP": 0, # Spain
    "UK": 1, # United Kingdom
    "DE": 2, # Germany
    "DK": 3, # Denmark
    "HU": 5, # Hungary
    "SE": 4, # Sweden
    "IT": 6, # Italy
    "PO": 7, # Poland
    "NE": 8 # Netherlands
    }

    list_of_countries = list(country_dict.keys())

    # add maximum suplus labels
    total_df['label'] = total_df.apply(lambda x: country_dict[max([(c, x[f"green_energy_{c}"] - x[f"load_{c}"]) for c in list_of_countries], 
                                     key=lambda item: item[1])[0]], axis=1)
    
    #fourth stage in data processing
    save_data(total_df, output_file)

    # report telemetry of the processing stage
    sum_up_events(SPAN_PROCESSOR)
    SPAN_PROCESSOR.shutdown()
    OLTP_EXPORTER.shutdown()
    
if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_path, args.output_file)