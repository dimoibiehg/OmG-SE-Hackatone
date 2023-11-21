from monitor import DATA_TRACER
import plotly.graph_objects as go
import os
import re


"""
output_path: shows the target folder that contains all fetched data

list_of_coutreis: indicates all possible countries, i.e., some countries may have no record in the fetced data

patterns: list of patterns for files which have been fetched for green energy and load of countries, resppectively.
          So, the first pattern should detect country code and its corresponding block. 
          The second code should detect the country code.
"""
def report_data_ingestion(output_path: str, list_of_countries: list[str], patterns=[r"gen_(\w{2})_([A-Za-z0-9]+).csv", r"load_(\w{2}).csv"]):
    
    files_addrs = os.listdir(output_path) # detecting all fetched files in the output_path
    
    # applying patters for detecting type of each file
    detected_patterns = [[] for i in range(len(patterns))]
    for  addr in files_addrs: 
        for i, pattern in enumerate(patterns):
            match = re.match(pattern, addr)
            if match:
                detected_patterns[i].append(match.groups())
    

    # extract set of existed blocks in fetched files
    list_of_all_blocks = []
    for gen in detected_patterns[0]:
        list_of_all_blocks.append(gen[1])

    list_of_all_blocks = sorted(list(set(list_of_all_blocks)))

    # initiate matrix of country_code-block_numbers by zeros
    country_block_pairs = []
    for block in list_of_all_blocks:
        country_block_pairs.append([])
        for country in list_of_countries:
            country_block_pairs[-1].append(0)
    
    # fill the matrix of the pairs sing read files
    for gen in detected_patterns[0]:
        country_idx = list_of_countries.index(gen[0])
        block_idx = list_of_all_blocks.index(gen[1])
        country_block_pairs[block_idx][country_idx] = 1

    is_fetched_load_country = [0 for x in list_of_countries]
    for gen in detected_patterns[1]:
        country_idx = list_of_countries.index(gen[0])
        is_fetched_load_country[country_idx] = 1

    # visualized detected data: green heatmap for exist]ed block of a contry and gray otherwise
    fig_gen = go.Figure(data=go.Heatmap(
                       z=country_block_pairs + [is_fetched_load_country],
                       x=list_of_countries,
                       y=list_of_all_blocks + ["Energy Consumption Load"],
                       colorscale=[[0, "rgb(200, 200, 200)"], [1, "rgb(0, 256, 0)"]],
                       xgap=5,
                       ygap=5,
                       showlegend=False,
                       hoverongaps = False))
    
    fig_gen.update_layout(xaxis_title="Country Code", yaxis_title="Block Number", title="Energy Files Status (Existed Files in Green, Gray otherwise)")
    fig_gen.show()

