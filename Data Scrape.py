from youtube_transcript_api import YouTubeTranscriptApi
from pytube import Playlist
from pytube import YouTube
from random import randint
from time import sleep
from tqdm import tqdm
import pyodbc as sql
import pandas as pd
import pickle
import string
import os

class NoTranscriptFound(Exception):
    pass


def ramsey_scrape(link, audio_location, transcript_location):
    con = sql.connect('''DRIVER={ODBC Driver 17 for SQL Server};
                  Server=ZANGORTH\HOMEBASE; DATABASE=RAMSEY; 
                  Trusted_Connection=yes;''')
                  
    query = 'SELECT * FROM RAMSEY.dbo.metadata'
    collected = pd.read_sql(query, con)
    con.close()
    
    i = collected.id.max() + 1
    
    if link in collected['link']:
        return 'Link already exists in database'
    
    yt = YouTube(link)
    
    if yt.channel_id != 'UC7eBNeDW1GQf2NJQ6G6gAxw':
        return 'Please only upload videos from The Ramsey Show - Highlights' 
    
    name = yt.streams[0].title
    name = name.translate(str.maketrans('', '', string.punctuation)).lower()
    
    keywords = '|'.join(yt.keywords)
    keywords = keywords.replace("'", "''").lower()
    
    con = sql.connect('''DRIVER={ODBC Driver 17 for SQL Server};
                      Server=ZANGORTH\HOMEBASE; DATABASE=RAMSEY; 
                      Trusted_Connection=yes;''')
                     
    csr = con.cursor()
    query = f'''
    INSERT INTO RAMSEY.dbo.streamlit 
    VALUES ({i}, '{name}', '{link}', '{yt.publish_date.strftime("%Y/%m/%d")}', '{keywords}', {yt.length}, {yt.rating}, {yt.views})'''
    
    csr.execute(query)
    csr.commit()
    con.close()
    
    yt.streams.filter(only_audio=True).first().download(f'{audio_location}/{name}')
    
    current_name = os.listdir(f'{audio_location}/{name}')[0]
    
    os.rename(f'{audio_location}\\{name}\\{current_name}', f'{audio_location}\\{i}.mp3')
    os.rmdir(f'{audio_location}\\{name}')
    
    transcript_link = link.split('?v=')[-1]
    
    try:
        transcript = YouTubeTranscriptApi.get_transcript(transcript_link)
        pickle.dump(transcript, open(f'{transcript_location}\\{i}.pkl', 'wb'))
    except NoTranscriptFound:
        pass
    except Exception:
        pass
    
    
def ramsey_scrape_all(audio_location, transcript_location):
    # Grab list of videos on the Ramsey channel
    videos = Playlist('https://www.youtube.com/watch?v=0JUw1agDjoA&list=UU7eBNeDW1GQf2NJQ6G6gAxw&index=2').video_urls
    videos = list(videos)
    
    # Grab list of videos that have already been recorded
    con = sql.connect('''DRIVER={ODBC Driver 17 for SQL Server};
                      Server=ZANGORTH\HOMEBASE; DATABASE=RAMSEY; 
                      Trusted_Connection=yes;''')
                      
    query = 'SELECT * FROM RAMSEY.dbo.metadata'
    collected = pd.read_sql(query, con)
    con.close()
    
    # Identify videos which need to be recorded
    videos = set(videos) - set(collected['link'])
    
    for video_link in tqdm(videos):
        ramsey_scrape(video_link, audio_location, transcript_location)
        
        sleep(randint(1, 3))