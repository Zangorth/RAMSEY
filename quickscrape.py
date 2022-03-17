from ramsey.ramsey import Scrape, encode_audio, upload
from multiprocessing import Pool
from pytube import Playlist
from time import sleep
import pyodbc as sql
import pandas as pd
import os

os.chdir(r'C:\Users\Samuel\Google Drive\Portfolio\Ramsey')

###############
# Scrape Data #
###############
personalities = {'ramsey': 'https://www.youtube.com/watch?v=j3NNJxYb9Sg&list=UU7eBNeDW1GQf2NJQ6G6gAxw',
                 'deloney': 'https://www.youtube.com/watch?v=ud3krvLD1bo&list=UU4HiMKM8WLcNbt9ae_XNRNQ',
                 'coleman': 'https://www.youtube.com/watch?v=PRNZ199_ax4&list=UU0tVfiyBpMOQLA3FAanPGJA',
                 'ao': 'https://www.youtube.com/watch?v=iE2yBe21-eA&list=PLPIXh_zvJ-4AEupWXKvlPQF2NFDQceoro',
                 'cruze': 'https://www.youtube.com/watch?v=-XU-xOg3pHM&list=UUt59W0ScV709iwy2h-oiulQ',
                 'wright': 'https://www.youtube.com/watch?v=zhp-1T1na8Q&list=UU1CHQyZ5-MTJzuSCvSVw_qg',
                 'kamel': 'https://www.youtube.com/watch?v=NXCsCMSWfiI&list=UUKFrkFOwmiXMuZtQJXuG5OQ'}

for personality in list(personalities.keys()):
    audio_location = f'Audio\\Audio Full\\{personality}'
    transcript_location = f'Audio\\Transcript\\{personality}'
    
    videos = Playlist(personalities[personality]).video_urls
    videos = list(videos)
    
    connection_string = ('DRIVER={ODBC Driver 17 for SQL Server};' + 
                         'Server=ZANGORTH;DATABASE=HomeBase;' +
                         'Trusted_Connection=yes;')
    con = sql.connect(connection_string)
    query = 'SELECT * FROM ramsey.metadata'
    collected = pd.read_sql(query, con)
    columns = pd.read_sql('SELECT TOP 1 * FROM ramsey.audio', con).columns
    con.close()
    
    videos = list(set(videos) - set(collected['link']))
    
    i = 1
    for video_link in videos:
        print(f'{personality}: {i}/{len(videos)}')
        i += 1
        
        try:
            new = Scrape(video_link, audio_location, transcript_location)
            metadata = new.metadata()
            new.audio()
            new.transcript()
            sleep(2)
            
        except Exception:
            metadata = ''
        
        if type(metadata) != str:
            iterables = new.iterables()
            iterables = [[personality, metadata.publish_date.item(), metadata.random_id.item(), sound[0], sound[1]] for sound in iterables]
            
            pool = Pool(28)
            audio_coding = pool.starmap(encode_audio, iterables)
            pool.close()
            pool.join()
            
            audio_coding = pd.concat(audio_coding)
            audio_coding.columns = columns
    
            metadata['seconds'] = audio_coding['second'].max()
            
            upload(metadata, 'ramsey', 'metadata')
            upload(audio_coding, 'ramsey', 'audio')
            
            