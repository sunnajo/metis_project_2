# Function for extracting album features
def get_album_features(title):
    '''
    For an album title, queries Spotify API for album and returns specified features of album
    and its artist in dictionary

    Parameters
    ----------
    title: album title as a string

    Returns
    ----------
    album_d: dictionary containing scraped information about album
    '''
    # Query
    try:
        q = sp.search(q=title, type='album', limit=10)['albums']['items'][0]
    except:
        return {'title': title,
        'artist': np.nan,
        'label': np.nan,
        'total_tracks': np.nan,
        'feat': np.nan,
        'genres': np.nan,
        'length': np.nan,
        'explicit': np.nan,
        'mode': np.nan,
        'avg_tempo': np.nan,
        'instrumentalness': np.nan,
        'album_sp_pop': np.nan,
        'artist_genres': np.nan,
        'artist_sp_pop': np.nan}
    
    # Extract features
    title = q['name']
    artist = q['artists'][0]['name']
    album_id = q['id']
    album_uri = q['uri']
    total_tracks = q['total_tracks']
    release_date = q['release_date']
    artist_uri = q['artists'][0]['uri']
    artist_id = q['artists'][0]['id']
    
    # Get more information about album
    album_info = sp.album(album_id)
    
    label = album_info['label']
    genres = album_info['genres']
    sp_pop = int(album_info['popularity'])
    tracks = album_info['tracks']['items']
    
    tracks_l = [get_track_features(track) for track in tracks]

    # Compile track information to determine album features
    album_length = 0
    album_exp = 0
    album_mode = 0
    album_tempo = 0
    album_inst = 0
    
    for track_d in tracks_l:
        album_length += track_d['length']
        album_feat = get_album_feat(track_d)
        album_exp += track_d['exp']
        album_mode += track_d['mode']
        album_tempo = np.mean(track_d['tempo'])
        album_inst += track_d['inst']
    
    # Get additional artist information
    artist_info = sp.artist(artist_id)
    artist_genres = artist_info['genres']   #Proxy for album genres if missing?
    artist_sp_pop = artist_info['popularity']
    
    # Create dictionary for each album
    album_d = {
        'title': title,
        'artist': artist,
        'label': label,
        'total_tracks': total_tracks,
        'feat:': album_feat
        'genres': genres,
        'length': album_length,
        'explicit': album_exp,
        'mode': album_mode,
        'avg_tempo': album_tempo,
        'instrumentalness': album_inst,
        'album_sp_pop': sp_pop,
        'artist_genres': artist_genres,
        'artist_sp_pop': artist_sp_pop
    }
    
    return album_d


# Helper functions
def get_featuring(track):
    '''
    For a track of an album, compiles list of all artists featured on the track and returns
    number of featuring artists as an integer
    '''
    track_artists = [artist['name'] for artist in track['artists']]
    feat = int(len(track_artists) > 1)
    return feat


def get_album_feat(track_d):
    '''
    For a dictionary containing track features, compiles total number of artists featured on
    album and returns whether or not album features >1 other artist as boolean integer
    '''
    feat_tot = 0
    album_feat = 0
    feat_tot += track_d['feat']
    if feat_tot == 0:
        album_feat = 0
    else:
        album_feat = 1


def get_track_features(track):
    '''
    For a track of an album, returns dictionary containing information about its audio features
    '''
    # Audio features
    audio = sp.audio_features(track['id'])
    danceability = audio[0]['danceability']
    energy = audio[0]['energy']
    key = audio[0]['key']
    loudness = audio[0]['loudness']
    mode = audio[0]['mode']
    speechiness = audio[0]['speechiness']
    acousticness = audio[0]['acousticness']
    instrumentalness = audio[0]['instrumentalness']
    liveness = audio[0]['liveness']
    valence = audio[0]['valence']
    tempo = audio[0]['tempo']
    time_signature = audio[0]['time_signature']
    track_length = audio[0]['duration_ms']

    # Create dictionary for each track
    track_d = {
        'id': track['id'],
        'name': track['name'],
        'num': track['track_number'],
        'length': track_length/1000,
        'feat': get_featuring(track),
        'exp': int(track['explicit']),
        'mode': mode,
        'inst': instrumentalness,
        'tempo': tempo
    }

    return track_d