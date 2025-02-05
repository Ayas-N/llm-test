import pandas as pd 
import os


rows_list = []
df = pd.read_csv("agent_out/banksy.csv").transpose()
df.at['-1']
def append_row(df, row):
    return pd.concat([
                df, 
                pd.DataFrame([row], columns=row.index)]
           ).reset_index(drop=True)

for sheet in os.listdir("agent_out")[1:]:
    if sheet.endswith(".csv"):
        tmp_df = pd.read_csv(f"agent_out/{sheet}").transpose()
        print(tmp_df.values[1])
        df.loc[len(df)] = tmp_df.values[1]

print(df)