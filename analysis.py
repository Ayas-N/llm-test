import pandas as pd 
import os


rows_list = []

def load_sheet(folder):
    '''Given a folder, load all csv files for each clustering methods
    folder: str of folder name
    returns: A dataframe containing all LLM predictions for clustering categories'''
    df = pd.read_csv(f"{folder}/banksy.csv", index_col=0).transpose()

    for sheet in os.listdir(folder)[1:]:
        if sheet.endswith(".csv"):
            tmp_df = pd.read_csv(f"{folder}/{sheet}", index_col=0).transpose()
            df = pd.concat([df,tmp_df])

    df = df.iloc[:,:-1]
    df['Run Type'] = folder
    return df

def load_google_sheet():
    '''Loads the spatial clustering review information sheet from google Sheets
    returns: A dataframe containing all labels'''
    gsheetid = "1P1-Nw0i_MpLoE8he1H7ZT-acYd4jOgDPrKZBxR-L6dw"
    sheet_name = "Methods" 
    online_sheet = f"https://docs.google.com/spreadsheets/d/{gsheetid}/gviz/tq?tqx=out:csv&sheet={sheet_name}&range=A2:A,I2:CV"
    info = pd.read_csv(online_sheet)
    info.rename(columns={'Unnamed: 0': 'Algorithm'}, inplace=True)
    info = info.dropna(subset = ["Parameter testing"])
    return info 

agent = load_sheet("agent_out")
print(agent.shape)
# gpt = load_sheet("gpt_out")
# pdfs = load_sheet("pdf_out")
# search = load_sheet("search_out")
# truth = load_google_sheet()
