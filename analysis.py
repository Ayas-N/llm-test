import pandas as pd 
import os
from matplotlib import pyplot as plt
import numpy as np

def clean_file(filename: str) -> None:
    '''A function to clean up the output because Gemini is too dumb to answer in the specified output.
    folder: str of filename name
    returns: None'''
    with open(filename,'r') as file:
        lines = file.readlines()
    
    cleaned_lines = [line.strip("```") for line in lines if '```' not in line]
    with open(filename,'w') as file:
        file.write(''.join(cleaned_lines))

def load_sheet(folder: str, sim_no: int) -> pd.DataFrame:
    '''Given a folder, load all csv files for each clustering methods
    folder: str of folder name
    sim_no: int of simulation number
    returns: A dataframe containing all LLM predictions for clustering categories'''
    clean_file(f"sim{sim_no}/{folder}/banksy.csv")
    # Grab the first 85 columns, have to do this beceause somehow Gemini messed up the one job it had.
    df = pd.read_csv(f"sim{sim_no}/{folder}/banksy.csv", index_col= 0, skipinitialspace= True).transpose()
    df = df.iloc[:,:85]
    df.index = df.index.str.strip()

    for sheet in os.listdir(f"sim{sim_no}/{folder}")[1:]:
        try: 
            if sheet.endswith(".csv"):
                clean_file(f"sim{sim_no}/{folder}/{sheet}")
                tmp_df = pd.read_csv(f"sim{sim_no}/{folder}/{sheet}", index_col= 0, skipinitialspace= True).transpose()
                # If there isn't the right columns after all this, you are disqualified!
                if tmp_df.shape[1] < 85:
                    tmp_df = pd.DataFrame([["None"] * len(df.columns)], columns = df.columns,
                                       index = [sheet[:-4].upper()]
                                       )
                    tmp_df.index = tmp_df.index.str.strip()
                    df = pd.concat([df,tmp_df], axis = 0)
                    continue
                tmp_df = tmp_df.iloc[:,:85]
                tmp_df.index = tmp_df.index.str.strip()
                tmp_df.columns = df.columns 
                df = pd.concat([df,tmp_df], axis = 0)
        except Exception as e:
            print(e)
            print(f"Error in {folder}")
            print("Adding in:")
            print(tmp_df)
            print("Current")
            print(df)
            break
    

    if df.columns[-1] == "```":
        df = df.iloc[:,:-1]
    df['Source'] = folder
    return df

def load_google_sheet() -> pd.DataFrame:
    '''Loads the spatial clustering review information sheet from google Sheets
    returns: A dataframe containing all labels'''
    gsheetid = "1P1-Nw0i_MpLoE8he1H7ZT-acYd4jOgDPrKZBxR-L6dw"
    sheet_name = "Methods" 
    online_sheet = f"https://docs.google.com/spreadsheets/d/{gsheetid}/gviz/tq?tqx=out:csv&sheet={sheet_name}&range=A2:A,I2:Y,AA2:AE,AH2:AW,AX2:BI,BL2:BQ,BS2:BU,BW2:CO,CQ2:CR,CT2:CX"
    info = pd.read_csv(online_sheet, index_col = 0)
    info = info.dropna(subset = ["Parameter testing"])
    info['Source'] = "Truth"

    return info

def filter_and_compare(llm_df: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    '''Joins a method with a specified LLM category and the ground truth dataframe together and 
    performs a comparison, by calculating the number of matching terms.
    
    llm_df (df): Dataframe of LLM
    truth (df): Ground Truth dataframe
    returns final: A dataframe containing the correctness of each algorithm'''
    methods = ['agent_out', 'gpt_out', 'search_out', 'pdf_out']
    dfs = []
    for method in methods:
        agent_filter = llm_df[llm_df['Source'] == method]
        agent_filter['Algorithm'] =  agent_filter['Algorithm'].str.upper()
        truth['Algorithm'] = truth['Algorithm'].str.upper()

        truth_filter = truth[truth['Algorithm'].isin(agent_filter['Algorithm'])]
        agent_filter = agent_filter[agent_filter['Algorithm'].isin(truth['Algorithm'])]

        agent_filter = agent_filter.sort_values(by= 'Algorithm').reset_index(drop = True)
        truth_filter = truth_filter.sort_values(by= 'Algorithm').reset_index(drop = True)

        compare = agent_filter.eq(truth_filter)
        compare['Algorithm'] = agent_filter['Algorithm']
        compare['Correct_Terms'] = compare.drop(columns='Algorithm').sum(axis = 1)
        compare['Correct_Percent'] = 100* compare['Correct_Terms'] / compare.shape[1]
        compare['Method'] = method 
        dfs.append(compare)

    final = pd.concat(dfs)
    return final

def evaluate(sim_no: int) -> pd.DataFrame:
    '''Whole evaluation for entire pipeline for simulation i
    Returns: Dataframe'''
    agent = load_sheet("agent_out", sim_no)
    gpt = load_sheet("gpt_out", sim_no)
    pdfs = load_sheet("pdf_out", sim_no)
    search = load_sheet("search_out", sim_no)
    truth = load_google_sheet()
    truth = truth[~truth.index.duplicated(keep='first')]
    truth.reset_index(inplace= True)

    llm_df = pd.concat([agent, gpt, search, pdfs])
    llm_df.reset_index(inplace = True)
    llm_df.rename(columns = {"index":"Algorithm"}, inplace = True)
    truth.columns = llm_df.columns
    llm_df.columns.name = None
    print(filter_and_compare(llm_df, truth))
    return filter_and_compare(llm_df, truth)

for i in range(1,6):
    simulation = evaluate(i)
    simulation.to_csv(f"data/sim{i}.csv")