import flask
from flask import request
import pickle
import numpy as np
import math
import spotipy
import csv
import pandas as pd
import requests
from spotipy import util
from flask_bootstrap import Bootstrap
import json
import cnfg
import os
fileDir = os.path.dirname(os.path.realpath('__file__'))
filename = os.path.join(fileDir, '.spotify_config')
config = cnfg.load(filename)


application = flask.Flask(__name__)
Bootstrap(application)


with open("logit_thresh2.pkl","rb") as f:
    logit = pickle.load(f)
with open("scale_func2.pkl.pkl", "rb") as f:
    scaler = pickle.load(f)


unm = "spotify:user:1255765740"
scope = "streaming"

token = util.prompt_for_user_token(
    unm,
    scope,
    client_id=config["client_id"],
    client_secret=config["client_secret"],
    redirect_uri='http://localhost/')
sp = spotipy.Spotify(auth=token)

billboard_averages=[0.16698920089938407,
 0.6257383983572878,
 0.6843842710472258,
 0.007811957868583155,
 0.1881537166324437,
 -5.917666119096504,
 0.10680069815195076,
 123.75755893223837,
 0.48535006160164296]


@application.route("/")
def hello():
    return flask.render_template('index2.html')


@application.route("/predict", methods=["POST", "GET"])
def predict():

    x_input = ''
        # f_value = 0
    f_value = request.args.get("track_name", "levels - avicii")
    try:
      token = util.prompt_for_user_token(
    unm,
    scope,
    client_id=config["client_id"],
    client_secret=config["client_secret"],
    redirect_uri='http://localhost/')
      sp = spotipy.Spotify(auth=token)
      searching=sp.search(f_value)['tracks']['items'][0]['uri']
    except:
      return flask.render_template('404.html')
    features=[]
    song=sp.track(searching)
    song_name=song['name']
    artist_name=song['album']['artists'][0].get('name')
    release_date=song['album']['release_date']
    #image=song['album']['images'][0]['url']
    iframeplayer='https://open.spotify.com/embed?uri='+searching


    #Processing the query
    audiof=sp.audio_features(searching)[0]
    artist=song['album']['artists'][0].get('uri')

    
    features=([np.log(song['duration_ms']),
           len(song['album'].get('artists'))-1,
           audiof.get('acousticness'),
           audiof.get('danceability'),
           audiof.get('energy'),
           audiof.get('instrumentalness'),
           audiof.get('liveness'),
           audiof.get('loudness'),
           audiof.get('speechiness'),
           audiof.get('tempo'),
           audiof.get('valence'),
           1 if 'rap' in sp.artist(artist).get('genres') else 0,
           1 if 'edm' in sp.artist(artist).get('genres') else 0,
           1 if 'country' in sp.artist(artist).get('genres') else 0,
           1 if 'rock' in sp.artist(artist).get('genres') else 0,
           1 if 'house' in sp.artist(artist).get('genres') else 0,
           1 if 'hip hop' in sp.artist(artist).get('genres') else 0,
           1 if 'indie' in sp.artist(artist).get('genres') else 0,
           1 if 'jazz' in sp.artist(artist).get('genres') else 0,
           1 if song['explicit']== True else 0

           
          ])
    items=features[2:11]
    
    features=scaler.transform(np.array(features).reshape((1, -1)))

    x_input=features

    pred_probs = logit.predict_proba(x_input)
    
    prob_yes= math.ceil(pred_probs[-1][-1]*100)
    pred_str = ("""Likely""")if prob_yes >46 else """Unlikely"""
    color=("""#1ED760""") if prob_yes >46 else """#ff6961"""
    probability=f"""{prob_yes}%"""


  
    indicies=[round((x/y)*100,0) for x,y in zip(items,billboard_averages)]
    indicies=[x if x<350 else 350 for x in indicies]
    


    # Return a response with a json in it
    # flask has a quick function for that that takes a dict
    return flask.render_template('predictor2.html',
    Name=song_name,
    Artist=artist_name,
    Release=release_date,
    x_input=x_input,
    prediction=pred_str,
    probability=probability,
    indicies=indicies,
    colors=color,
    iframeplayer=iframeplayer)

@application.route("/artist", methods=["POST", "GET"])
def artist():
    token = util.prompt_for_user_token(
    unm,
    scope,
    client_id=config["client_id"],
    client_secret=config["client_secret"],
    redirect_uri='http://localhost/')
    sp = spotipy.Spotify(auth=token)

    artists=request.args.get("artist")
    searching=sp.search(q=artists,type='artist',market='us')['artists']['items'][0]['uri']
    picture=sp.artist(searching)['images'][0].get('url')
    followers=sp.artist(searching)['followers']['total']
    results = sp.artist_top_tracks(searching)['tracks']
    tracklist=[x['uri'] for x in results]
    

    song_details=[]
    features=[]
    song_details+=sp.tracks(tracklist)['tracks'] 
    features+= sp.audio_features(tracklist)
    y=list(range(len(features)))
    feat=list(zip(y,features))
    feat=pd.DataFrame(feat)
    song_df=pd.DataFrame(song_details)
    audio=feat[1].apply(pd.Series)
    audio=audio.drop(columns='duration_ms')
    audio=audio.sort_index(axis=1)
    song_df.drop(columns=['available_markets','disc_number','external_urls',\
                      'external_ids','href','is_local','preview_url','track_number','type'],inplace=True)
    
    album1=song_df['album'].apply(pd.Series)
    album1=album1['release_date']
    song_df['collabs']=[len(x)-1 for x in song_df['artists']]
    song_df['artist_name']=[x[0].get('name') for x in song_df['artists']]
    song_df['artist_uri']=[x[0].get('uri') for x in song_df['artists']]
    song_info=pd.concat([song_df, album1], axis=1, join_axes=[song_df.index])

    spotify_info=pd.concat([song_info, audio], axis=1, join_axes=[song_info.index])
    names=spotify_info.name.tolist()
    names=[x if len(x)<18 else str(x[0:15]+"...") for x in names]
    spotify_info=spotify_info.drop(columns=['album','artists','id','popularity','uri','artist_name','artist_uri','release_date',\
      'analysis_url','duration_ms.1','id.1','mode','time_signature','track_href','type','uri.1','name','key',])
    
    spotify_info['is_rap']= 1 if 'rap' in sp.artist(searching).get('genres') else 0
    spotify_info['is_edm']= 1 if 'edm' in sp.artist(searching).get('genres') else 0
    spotify_info['is_country']=1 if 'country' in sp.artist(searching).get('genres') else 0
    spotify_info['is_rock']=1 if 'rock' in sp.artist(searching).get('genres') else 0
    spotify_info['is_house']=1 if 'house' in sp.artist(searching).get('genres') else 0
    spotify_info['is_hiphop']=1 if 'hip hop' in sp.artist(searching).get('genres') else 0
    spotify_info['is_indie']=1 if 'indie' in sp.artist(searching).get('genres') else 0
    spotify_info['is_jazz']=1 if 'jazz' in sp.artist(searching).get('genres') else 0
    spotify_info['is_explicit']= [1 if x==True else 0 for x in spotify_info['explicit']]
    spotify_info= spotify_info.fillna(value=0)
    spotify_info=spotify_info.drop(columns='explicit')
    spotify_info.duration_ms=spotify_info.duration_ms+1
    spotify_info.duration_ms=np.log(spotify_info.duration_ms)
    
    playlists=scaler.transform(spotify_info)
    
    
    pred_probs = logit.predict_proba(playlists)
    prob_yes= [math.ceil(x[-1]*100) for x in pred_probs]
    composite_score=int(np.average(prob_yes,weights=[.2,.15,.15,.1,.1,.08,.08,.08,.08,.08]))

    # pred_str = ("""Likely""")if composite_score >46 else """Unlikely"""
    colors=("""#1ED760""") if composite_score >=50.0 else """#ff6961""" 
    followers="{:,}".format(followers)


    
    iframeplayer='https://open.spotify.com/embed?uri='+searching
    return flask.render_template('artist2.html',
      Artist=artists,
      composite_score=composite_score,
      colors=colors,
      iframeplayer=iframeplayer,
      names=names,
      prob_yes=prob_yes,
      picture=picture,
      followers=followers
      )
# Start the server, continuously listen to requests.
# We'll have a running web app!

# For local development:


# For public web serving:
if __name__ == "__main__":
  application.debug = True
  application.run()
