from youtube_transcript_api import YouTubeTranscriptApi
from pytube import Playlist
from pytube import YouTube
from random import randint
from time import sleep
import pyodbc as sql
import pandas as pd
import pickle
import string
import os

class NoTranscriptFound(Exception):
    pass


folder = r'C:\Users\Samuel\Audio'

videos = Playlist('https://www.youtube.com/watch?v=0JUw1agDjoA&list=UU7eBNeDW1GQf2NJQ6G6gAxw&index=2').video_urls
videos = list(videos)


con = sql.connect('''DRIVER={ODBC Driver 17 for SQL Server};
                  Server=ZANGORTH\HOMEBASE; DATABASE=RAMSEY; 
                  Trusted_Connection=yes;''')
                  
query = 'SELECT * FROM RAMSEY.dbo.metadata'
collected = pd.read_sql(query, con)
con.close()

videos = set(videos) - set(collected['link'])

title, keywords, length, rating, views = [], [], [], [], []

i = collected.id.max()
for video_link in videos:
    i += 1
    print(f'Progress: {i}/{len(videos) + collected.id.max()}')
    
    yt = YouTube(video_link)
    
    name = yt.streams[0].title
    name = name.translate(str.maketrans('', '', string.punctuation)).lower()
    
    if yt.length > 900:
        con = sql.connect('''DRIVER={ODBC Driver 17 for SQL Server};
                          Server=ZANGORTH\HOMEBASE; DATABASE=RAMSEY; 
                          Trusted_Connection=yes;''')
                         
        csr = con.cursor()
        query = f'''
        INSERT INTO RAMSEY.dbo.metadata 
        VALUES ({i}, '{name}', '{video_link}', '{yt.publish_date.strftime("%Y/%m/%d")}', '{keywords}', {yt.length}, {yt.rating}, {yt.views})'''
        
        csr.execute(query)
        csr.commit()
        con.close()
        
    
    elif not os.path.exists(f"{folder}/{name}.mp3") and yt.length <= 900:
        keywords = '|'.join(yt.keywords)
        keywords = keywords.replace("'", "''").lower()
        
        con = sql.connect('''DRIVER={ODBC Driver 17 for SQL Server};
                          Server=ZANGORTH\HOMEBASE; DATABASE=RAMSEY; 
                          Trusted_Connection=yes;''')
                         
        csr = con.cursor()
        query = f'''
        INSERT INTO RAMSEY.dbo.metadata 
        VALUES ({i}, '{name}', '{video_link}', '{yt.publish_date.strftime("%Y/%m/%d")}', '{keywords}', {yt.length}, {yt.rating}, {yt.views})'''
        
        csr.execute(query)
        csr.commit()
        con.close()
        
        yt.streams.filter(only_audio=True).first().download(f'{folder}/{name}')
        
        os.rename(f"{folder}/{name}/{os.listdir(f'{folder}/{name}')[0]}", f'{folder}/Audio Full/{i}.mp3')
        os.rmdir(f'{folder}/{name}')
        
        link = video_link.split('?v=')[-1]
        
        try:
            transcript = YouTubeTranscriptApi.get_transcript(link)
            pickle.dump(transcript, open(f'C:\\Users\\Samuel\\Audio\\Transcript\\{i}.pkl', 'wb'))
        except NoTranscriptFound:
            pass
        except Exception:
            pass
        
        
        
    sleep(randint(1, 3))