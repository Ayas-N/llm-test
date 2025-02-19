import pandas as pd 
import os
from matplotlib import pyplot as plt

def load_sheet(folder):
    '''Given a folder, load all csv files for each clustering methods
    folder: str of folder name
    returns: A dataframe containing all LLM predictions for clustering categories'''
    df = pd.read_csv(f"{folder}/banksy.csv", index_col= 0, skipinitialspace= True).transpose()
    df.index = df.index.str.strip()

    for sheet in os.listdir(folder)[1:]:
        try: 
            if sheet.endswith(".csv"):
                tmp_df = pd.read_csv(f"{folder}/{sheet}", index_col= 0, skipinitialspace= True).transpose()
                tmp_df.index = tmp_df.index.str.strip()
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

def load_google_sheet():
    '''Loads the spatial clustering review information sheet from google Sheets
    returns: A dataframe containing all labels'''
    gsheetid = "1P1-Nw0i_MpLoE8he1H7ZT-acYd4jOgDPrKZBxR-L6dw"
    sheet_name = "Methods" 
    online_sheet = f"https://docs.google.com/spreadsheets/d/{gsheetid}/gviz/tq?tqx=out:csv&sheet={sheet_name}&range=A2:A,I2:Y,AA2:AE,AH2:AW,AX2:BI,BL2:BQ,BS2:BU,BW2:CO,CQ2:CR,CT2:CX"
    info = pd.read_csv(online_sheet, index_col = 0)
    info = info.dropna(subset = ["Parameter testing"])
    info['Source'] = "Truth"

    return info

agent = load_sheet("agent_out")
gpt = load_sheet("gpt_out")
pdfs = load_sheet("pdf_out")
search = load_sheet("search_out")
truth = load_google_sheet()
truth = truth[~truth.index.duplicated(keep='first')]
truth.reset_index(inplace= True)

llm_df = pd.concat([agent, gpt, search, pdfs])
llm_df.reset_index(inplace = True)
llm_df.rename(columns = {"index":"Algorithm"}, inplace = True)
llm_df.columns.name = None

llm_df.columns = truth.columns
agent_filter = llm_df[llm_df['Source'] == "agent_out"]

truth_filter = truth[truth['Algorithm'].isin(agent_filter['Algorithm'])]
agent_filter = agent_filter[agent_filter['Algorithm'].isin(truth['Algorithm'])]

agent_filter = agent_filter.sort_values(by= 'Algorithm').reset_index(drop = True)
truth_filter = truth_filter.sort_values(by= 'Algorithm').reset_index(drop = True)

compare = agent_filter.eq(truth_filter)
compare['Algorithm'] = agent_filter['Algorithm']
compare['Correct_Terms'] = compare.drop(columns='Algorithm').sum(axis = 1)
compare['Correct_Percent'] = 100* compare['Correct_Terms'] / compare.shape[1]
print(compare)


