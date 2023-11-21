import pandas as pd
import datetime
from opentelemetry.sdk.trace.export import BatchSpanProcessor
import warnings

"""
rename columns, for example, "Load" column to "qunatity" 
"""
def unify_column_names(df: pd.DataFrame, transfers=[["Load", "quantity"]]):
    # rankings_pd.rename(columns = {'test':'TEST'}, inplace = True)
    col_names = list(df.columns)
    for x in transfers:
        if(x[0] in col_names):
            df = df.rename(columns = {x[0]:x[1]})
    return df

"""
target_cols: indicates columns that should be considered for filling missing data
"""
def filling_missing_data(df: pd.DataFrame, method="linear", direction="both", inplace=True):
    # apply mean of neighbors for missing data
    return  df.interpolate(method=method, limit_direction=direction)

def make_correct_data_types(df: pd.DataFrame, datetime_cols=["StartTime", "EndTime"], 
                            datetime_pattern='%Y-%m-%dT%H:%M%zZ', float_cols=["quantity"] ):
    for col in datetime_cols:
        df[col] = df.apply(lambda x: datetime.datetime.strptime(x[col], datetime_pattern) , axis=1)

    for col in float_cols:
        df[col] = df[col].astype(float)
    
    return df

def remove_out_of_time_limit_data(df: pd.DataFrame, start_time:datetime.datetime, end_time:datetime.datetime,
                                  start_col = "StartTime", end_col="EndTime"):
    mask = ((df['StartTime'] >= start_time) & (df[start_col] < end_time) & (df[end_col] <= end_time))
    return df.loc[mask, :].reset_index()

def fill_zero_not_exist_timesteps(df: pd.DataFrame, start_time:datetime.datetime, end_time:datetime.datetime, 
                                  start_col = "StartTime"):
    idx = pd.date_range(start=start_time, end=end_time-datetime.timedelta(hours=1), freq='1h')
    df = df.reindex(idx, fill_value=0)
    return df

"""
Groups data in df in timestep which is determined by group_freq

start_col: grouping data is based on this column
quality_col: indictae the dolumn that should be sumaarized after grouping data 
"""
def group_data_points(df: pd.DataFrame, start_col="StartTime", quantity_col="quantity",  gropup_freq="1H"):
    new_data: pd.DataFrame = df.groupby(pd.Grouper(key=start_col, freq=gropup_freq)).agg({quantity_col:"sum"})
    new_data.columns = [quantity_col]   
    new_data.index.names = ["index"]
    return new_data

def unify_energy_units(x, qunatity_column_name, 
                         start_time_col_name = "StartTime", 
                         end_time_col_name = "EndTime", 
                         target_unit = "MAW", energy_unit_column_name = "UnitName"):

    # The list of exisiting enery units comes from the released document by ENTSOE in the following document:
    # https://eepublicdownloads.entsoe.eu/clean-documents/pre2015/resources/Transparency/MoP%20Ref05%20-%20EMFIP-1-gl-market-document_V3R0-2014-01-24.pdf 
    # At the moment (i.e., Nov 20 2023) MAW and MWH have been identified.
    if(x[energy_unit_column_name] == target_unit):
        return [x[energy_unit_column_name], x[qunatity_column_name]]
    elif(x[energy_unit_column_name] == "MWH"):
        return [x[energy_unit_column_name], x[qunatity_column_name]/ 
                                            ((x[end_time_col_name] - x[start_time_col_name]).seconds/3600)]
    else:
        raise Exception("An non-identified 'energy unit' has been detected!")

def make_energy_units_unified(df, qunatity_column_name, 
                                start_time_col_name = "StartTime", 
                                end_time_col_name = "EndTime", 
                                energy_unit_column_name = "UnitName", 
                                target_unit="MAW"):

    mask = (df[energy_unit_column_name] != target_unit)
    df.loc[mask, [energy_unit_column_name, qunatity_column_name]] = df.loc[mask, :].apply(lambda x:  unify_energy_units(x, qunatity_column_name, 
                                                                                                                        start_time_col_name = start_time_col_name, 
                                                                                                                        end_time_col_name = end_time_col_name, 
                                                                                                                        target_unit = target_unit, 
                                                                                                                        energy_unit_column_name = energy_unit_column_name
                                                                                                                        ))
    return df


"""
Summarize and print added telemetry events to report some statistics for the matter of monitoring of the status of the system
like the number of missing values in each file.
"""
def sum_up_events(span_processor: BatchSpanProcessor):
    
    print("#################")
    print("number of missing values in the target columns:")
    for span in span_processor.queue:
            val = span.events[0].attributes.get("value")
            country_code = span.events[0].attributes.get("country_cdoe")
            if(span.events[0].name.endswith("gen")):
                block_code = span.events[0].attributes.get("country_cdoe")
                print(f"gen {country_code}-{block_code}: {val}")
            elif(span.events[0].name.endswith("load")):
                print(f"load {country_code}: {val}")
            else:
                warnings.warn("a telemtry which is not fitted into the patterns!")
    print("#################")


def find_surplus_country(row, list_of_countries, country_dict):
    diffs = []
    for c in list_of_countries:
        gen_energy_col = "green_energy_NL" # f"green_energy_{c}"
        load_col = f"load_{c}"
        diffs.append((c, (row[gen_energy_col] - row[load_col])))
    
    max_item = max(diffs, key=lambda item: item[1])
    return max_item[0]