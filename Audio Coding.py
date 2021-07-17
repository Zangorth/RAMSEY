import multiprocessing as mp
import pyodbc as sql
import pandas as pd
import warnings
import sys
import os

os.chdir(r'C:\Users\Samuel\Audio\Audio Full')
sys.path.append(r'C:\Users\Samuel\Google Drive\Portfolio\Ramsey')
from extract_audio import extract_audio
warnings.filterwarnings('ignore')


con = sql.connect('''DRIVER={ODBC Driver 17 for SQL Server};
                  Server=ZANGORTH\HOMEBASE; DATABASE=RAMSEY; 
                  Trusted_Connection=yes;''')
                  
query = 'SELECT DISTINCT id FROM RAMSEY.dbo.AudioCoding'
completed = pd.read_sql(query, con)
completed = completed['id'].tolist()
con.close()

incomplete = os.listdir()
incomplete = list(set([int(file.replace('.mp3', '')) for file in incomplete]) - set(completed))
incomplete = [f'{file}.mp3' for file in incomplete]

for i in range(len(incomplete)):
    print(f'Progress: {i}/{len(incomplete)}')
    extract_audio(incomplete[i])
