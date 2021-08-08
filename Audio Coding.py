from sqlalchemy import create_engine
from pydub import AudioSegment
import multiprocessing as mp
import pyodbc as sql
import pandas as pd
import warnings
import urllib
import sys
import os

sys.path.append(r'C:\Users\Samuel\Google Drive\Portfolio\Ramsey')
from ramsey_helpers import extract_audio
warnings.filterwarnings('ignore')

os.chdir(r'C:\Users\Samuel\Audio\Audio Full')


con = sql.connect('''DRIVER={ODBC Driver 17 for SQL Server};
                  Server=ZANGORTH\HOMEBASE; DATABASE=RAMSEY; 
                  Trusted_Connection=yes;''')
                  
query = 'SELECT DISTINCT id FROM RAMSEY.dbo.AudioCoding'
completed = pd.read_sql(query, con)
completed = completed['id'].tolist()

columns = pd.read_sql('Select TOP 1 * FROM RAMSEY.dbo.AudioCoding', con)
columns = columns.columns
con.close()

incomplete = os.listdir()
incomplete = list(set([int(file.replace('.mp3', '')) for file in incomplete]) - set(completed))
incomplete = [f'{file}.mp3' for file in incomplete]

for i in range(len(incomplete)):
    print(f'Progress: {i}/{len(incomplete)}')
    
    sound = AudioSegment.from_file(incomplete[i])
    iterables = [[int(incomplete[i].replace('.mp3', '')), cut, sound[cut*1000:cut*1000+1000]] for cut in range(int(round(len(sound)/1000, 0)))]
    
    with mp.Pool(10) as pool:
        out = pool.map(extract_audio, iterables)
        
    out = [clip for clip in out if clip != []]
    panda = pd.DataFrame(out, columns=columns)
        
    conn_str = (
        r'Driver={SQL Server};'
        r'Server=ZANGORTH\HOMEBASE;'
        r'Database=RAMSEY;'
        r'Trusted_Connection=yes;'
    )
    con = urllib.parse.quote_plus(conn_str)
    
    engine = create_engine(f'mssql+pyodbc:///?odbc_connect={con}')
    
    panda.to_sql(name='AudioCoding', con=engine, schema='dbo', if_exists='append', index=False)
    
