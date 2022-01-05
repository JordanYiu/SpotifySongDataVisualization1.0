# %%
import spotipy as sp
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import numpy as np
import time

# %%
clientid = ";)"
clientsecret = ";)"
redirecturi = "http://127.0.0.1:9090"
scope1 = "user-top-read"

client_cred = f"{clientid}:{clientsecret}"
sp = sp.Spotify(auth_manager=SpotifyOAuth(client_id=clientid,
                client_secret=clientsecret, redirect_uri=redirecturi, scope=scope1))


# %%
results = sp.current_user_top_artists()

# %%
# Get artist name
def get_artist_name(artist):
    artist_name = []
    for i in range(len(artist["items"])) :
        artist_name.append(artist["items"][i]["name"])
    return artist_name


# %%
# Get artist URI
def get_artist_uri(artist):
    artist_uri = []
    for i in range(len(artist["items"])):
        artist_uri.append(artist["items"][i]["uri"])
    return artist_uri


# %%
artist_uris = get_artist_uri(results)
artist_uris = [s.replace("spotify:artist:","") for s in artist_uris]

artist = get_artist_name(results)

# %%
# artist_uris

# %%
# ONLY Artists and URIs in a table for later reference/table building.
# Average song features across all albums maybe or something idk
df1 = pd.DataFrame()
df1["Artist"] = artist
df1["URI"] = artist_uris

# %%
# Get artist albums from artist URI
def get_artist_album(uri):
    albums = []
    for k in range(len(uri)): # For loop to get artists, then album
        # Get albums from artist URIs
        albums.append(sp.artist_albums(uri[k], album_type="album", country="US"))
    return albums


# %%
# JSON blob of album data
tempalbums = get_artist_album(artist_uris)

# %%
# # Test cell to get album URIs. This took too long for me to figure out
# for j in range(len(tempalbums)) :
#     idk = []
#     for i in range(len(tempalbums[j]["items"])) :
#         idk.append(tempalbums[j]["items"][i]["uri"])
# #print(tempalbums[j]["items"])#["items"][13]["uri"]


# %%
# Get album URI. Input = tempalbums

album_name = []
album_uri = []
def get_album_uri(album) :
    for k in range(len(album)) :
        for i in range(len(album[k]["items"])):
            album_name.append(album[k]["items"][i]["name"])
            album_uri.append(album[k]["items"][i]["uri"])
    return album_uri


# %%
album_uris = get_album_uri(tempalbums)

# %%
# Get tracks for each song in each album. Input = album_uris
def get_track_from_album(albumuri):
    tracks = []
    for track in range(len(albumuri)):
        tracks.append(sp.album_tracks(albumuri[track], market = "US"))
    return tracks

# %%
tracks = get_track_from_album(album_uris)

# %%
# Get track URIs, artist, and album info into lists from track info for audio_features. Input = tracks.
def get_track_uri(tracks) :
    track_uris = []
    for k in range(len(tracks)) :
        for i in range(len(tracks[k]["items"])) :
            track_uris.append(tracks[k]["items"][i]["uri"])
    return track_uris

# %%
def get_track_name(trackuri) :
    track_names = []
    for k in range(len(trackuri)):
        for i in range(len(trackuri[k]["items"])) :
            track_names.append(trackuri[k]["items"][i]["name"])
    return track_names

# %%
track_uris = get_track_uri(tracks)
track_uris = list(set(track_uris))
track_uris

# %%
test = sp.audio_features("3u2TWIOpWwFEndbTyDLWu2")
test

# %%
# test = sp.track("5ohsTkUXwgGfR1t9YEaM7W")
# test["name"] #["artists"][0]["name"]

# %%
track_names = get_track_name(tracks)
track_names

# %%
# Get audio features. Input = track_uris
def get_audio_features(uris) :
    # Time delay to avoid flooding API
    # request_count = 0
    # sleep_min = 2
    # sleep_max = 5
    # start_time = time.time()

    audio_features = {}
    # features_list = ["acousticness", "danceability", "energy", "instrumentalness", "liveness", "loudness", "speechiness", "tempo", "valence"]

    # Nested dictionary for each song, via URI
    for j in range(len(uris)) :
        audio_features[track_uris[j]] = {}
    for track in range(len(uris)):
        # uris[track] = single song URI
        audio_features[track_uris[track]]["artist"] = []
        audio_features[track_uris[track]]['name'] = []
        audio_features[track_uris[track]]['acousticness'] = []
        audio_features[track_uris[track]]['danceability'] = []
        audio_features[track_uris[track]]['energy'] = []
        audio_features[track_uris[track]]['instrumentalness'] = []
        audio_features[track_uris[track]]['liveness'] = []
        audio_features[track_uris[track]]['loudness'] = []
        audio_features[track_uris[track]]['speechiness'] = []
        audio_features[track_uris[track]]['tempo'] = []
        audio_features[track_uris[track]]['valence'] = []
        audio_features[track_uris[track]]['time_signature'] = []
        audio_features[track_uris[track]]['key'] = []
        
    # Add a song's features to it's individual dictionary
    for i in range(len(uris)) :
        features = sp.audio_features(uris[i])
        artist = sp.track(uris[i])
        # If Spotify has song data, add the data. Also add the song name.
        if features[0] is not None:
            audio_features[track_uris[i]]["artist"].append(artist["artists"][0]["name"])
            audio_features[track_uris[i]]["name"].append(artist["name"])
            audio_features[track_uris[i]]["acousticness"].append(features[0]["acousticness"])
            audio_features[track_uris[i]]["danceability"].append(features[0]["danceability"])
            audio_features[track_uris[i]]["energy"].append(features[0]["energy"])
            audio_features[track_uris[i]]["instrumentalness"].append(features[0]["instrumentalness"])
            audio_features[track_uris[i]]["liveness"].append(features[0]["liveness"])
            audio_features[track_uris[i]]["loudness"].append(features[0]["loudness"])
            audio_features[track_uris[i]]["speechiness"].append(features[0]["speechiness"])
            audio_features[track_uris[i]]["tempo"].append(features[0]["tempo"])
            audio_features[track_uris[i]]["valence"].append(features[0]["valence"])
            audio_features[track_uris[i]]["time_signature"].append(features[0]["time_signature"])
            audio_features[track_uris[i]]["key"].append(features[0]["key"])
            # request_count += 1
        # If Spotify DOES NOT have the song data, skip. 
        if features is None :
            continue

        # # Delay timer to avoid flooding API. Don't think this is entirely necessary
        # if request_count % 5 == 0 :
        #     print(str(request_count))
        #     time.sleep(np.random.uniform(sleep_min, sleep_max))
        #     print("Elapsed Time: {} seconds".format(time.time() - start_time))
    return audio_features
    


# %%
# testfeatures = sp.audio_features("1IUIsNQCMkRXcBnFR9odXF")
# testfeatures


# %%
audio_features_dic = get_audio_features(track_uris)
audio_features_dic

# %%
audio_features_df = pd.DataFrame.from_dict(audio_features_dic, orient="index")

# %%
columns = ['artist', 'name', 'acousticness', 'danceability', 'energy',
           'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 'time_signature', 'key']
for i in range(len(columns)):
    audio_features_df[columns[i]] = audio_features_df[columns[i]].astype(str)
    audio_features_df[columns[i]] = audio_features_df[columns[i]].str.strip('[]')


# %%
audio_features_df

# %%
audio_features_df.to_csv("audiodata.csv")


