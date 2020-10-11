'''
This module contains functions for web scraping information about albums and corresponding
artists from Album of the Year's list of albums using Beautiful Soup
'''
# Functions for scraping album page
def get_album_links(page):
    '''
    For a page extension, navigates to desired page of list, gets and parses HTML of webpage,
    scrapes links for individual album pages, and returns list of links
    '''
    # Get & parse HTML
    base_url = 'https://www.albumoftheyear.org/ratings/user-highest-rated/all/hip-hop/'
    url = base_url + page
    response = requests.get(url)
    page = response.content
    soup = BeautifulSoup(page, 'lxml')
    
    # Create list of links to album pages for each page of master list
    album_links = [h2.find('a').get('href') for h2 in soup.find_all('h2', class_='albumListTitle')]
    return album_links


def scrape_album(link):
    '''
    For an extension, navigates to desired webpage for an album, gets and parses HTML of webpage,
    scrapes specified information/features, and returns features in dictionary

    Parameters
    ----------
    link: extension for webpage

    Returns
    ----------
    album_dict: dictionary containing scraped information about album
    '''
    # Getting and parsing HTML
    base_url = 'https://www.albumoftheyear.org'
    url = base_url + linkhttps://www.albumoftheyear.org/ratings/user-highest-rated/all/hip-hop/12/?r=50
    response = requests.get(url)
    page = response.content
    soup = BeautifulSoup(page, 'lxml')
    
    # Scraping features
    # Clean strings
    title = soup.find('div', class_='albumTitle').find('span').text.strip()
    artist = soup.find('div', class_='artist').find('span').find('a').text.strip()
    critic_score = get_critic_score(soup)
    release_date = get_release_date(soup)
    platforms = get_platforms(soup)
    album_format = get_format(soup)
    labels = get_labels(soup)
    major_label = get_major_label(labels) 
    genres = get_genres(soup)
    user_score = int(soup.find('div', class_='albumUserScore').text.strip())
    user_ratings = get_user_ratings(soup)
    num_tracks = get_track_num(soup)
    featuring = get_featuring(soup)
    artist_link = soup.find('div', class_='artist').find('a').get('href')
    
    # Create dictionary
    album_dict = {
        'title': title,
        'artist': artist,
        'release_date': release_date,
        'format': album_format,
        'major_label': major_label,
        'genres': genres,
        'num_tracks': num_tracks,
        'features_others': featuring,
        'streaming': platforms,
        'critic_score': critic_score,
        'user_score': user_score,
        'num_user_ratings': user_ratings
    }
    
    return album_dict

# Helper functions for scraping album page
def get_user_ratings(soup):
    '''
    Returns number of user ratings for an album as integer
    '''
    user_ratings = soup.find('div', class_='albumUserScoreBox').find('div', class_='text numReviews').find('a').find('strong').text
    user_ratings = user_ratings.replace(',', '')
    return int(user_ratings)


def get_release_date(soup):
    '''
    Returns release date for an album as object
    '''
    for div in soup.find_all('div', class_='detailRow'):
        if re.search('[Rr]elease', div.span.text):
            date = div.text.split('/')[0].split()
            date = ' '.join(date)  
    
    return date


def get_format(soup):
    '''
    Returns format of an album as a boolean integer:
    0 = non-LP, 1 = LP
    '''
    f = 0

    for div in soup.find_all('div', class_='detailRow'):
        if re.search('[Ff]ormat', div.span.text):
            f_str = div.text.strip()
            f_str = f_str.split(' /')[0].strip()
            if f_str == 'LP':
                f = 1
    
    return f


def get_labels(soup):
    '''
    Returns list of name(s) of label(s) for album as string
    '''
    labels_str = [div.text.strip() for div in soup.find_all('div', class_='detailRow') if re.search('[Ll]abel', div.span.text)]
    labels = [l.split(' /')[0].strip() for l in labels_str]
    
    return labels


def get_major_label(labels):
    '''
    For a list of an album's label(s), returns whether or not album is from
    major record label as boolean integer

    Parameters
    ----------
    labels: list of label(s) for album

    Returns
    ----------
    boolean integer: 0 = non-major, 1 = major
    '''
    # List of major record labels (from https://www.loc.gov/programs/national-recording-preservation-board/resources/major-record-labels/)
    major_labels = ['A&M Records', 'Abet', 'Aftermath', 'Alligator Records', 'Arista', 'Arista Nashville',
                    'Aretemis', 'Astralwerks', 'Asylum Records', 'Atlantic', 'Bad Boy', 'Blue Note',
                    'Boosweet Records', 'Capitol', 'Cedarmont Kids', 'Columbia', 'Curb', 'Def Jam', 'Delve in Dreams',
                    'Deutsche Grammophon', 'Disney Records', 'Doggy Style', 'Elektra', 'EMI CMG', 'Epic',
                    'Essential Records', 'Geffen Records', 'Hip-O Records', 'Hollywood Records', 'Immortal',
                    'Interscope', 'Island', 'J Records', 'Lava', 'Legacy', 'Maverick', 'MCA Nashville',
                    'Mercury Nashville', 'Motown', 'Mute', 'Nonesuch', 'Priority', 'RCA', 'Red Ink Records',
                    'Reprise Records', 'Republic Records', 'Reunion Records', 'Rhino', 'Roadrunner', 'Shady', 'Sire',
                    'Smithsonian Folkways', 'Sony BMG', 'Sony Latin', 'Sony Masterworks', 'Star Trak', 'Universal',
                    'Universal Classics', 'Universal Latin', 'Verve', 'Virgin', 'Warner Brothers',
                    'Warner Brothers Latin', 'Warner Brothers Nashville', 'Word']
    
    major = 0
    
    for label in labels:
        if label in major_labels:
            major = 1
    
    return major


def get_genres(soup):
    '''
    Returns genre of album as string
    '''
    for div in soup.find_all('div', class_='detailRow'):
        if re.search('[Gg]enres', div.span.text):
            genre = div.a.text.strip()
    
    return genre


def get_track_list(soup):
    '''
    Returns list of tracks of album
    '''
    tracks = soup.find('div', class_='trackList').ol.findChildren()
    return tracks


def get_track_num(soup):
    '''
    Returns number of tracks of album as integer
    '''
    try:
        tracks = soup.find('div', class_='trackList').ol.findChildren()
    except:
        return np.nan

    num_tracks = 0
    for li in tracks:
        num_tracks += 1
    
    return num_tracks


def get_featuring(soup):
    '''
    Returns whether album features another artist as boolean integer:
    0 = no feature, 1 = featuring
    '''
    try:
        tracks = soup.find('div', class_='trackList').ol.findChildren()
    except:
        return np.nan

    feat = 0
    for li in tracks:
        if re.search('[Ff][A-Za-z]*t\.', li.text):
            feat = 1
    
    return feat


def get_platforms(soup):
    '''
    Returns whether or not album is available on any streaming platform
    as boolean integer: 0 = none, 1 = available
    '''
    online = ['Amazon', 'iTunes', 'Apple Music', 'Spotify']
    platforms = [a.get('title') for a in soup.find('div', class_='buyButtons').find_all('a')]
    
    return int(any(item in platforms for item in online))


def get_critic_score(soup):
    '''
    Returns critic score for album as integer
    '''
    try:
        critic_score = soup.find('div', class_='albumCriticScore').find('span').find('a').text.strip()
    except:
        return np.nan
    
    return int(critic_score)


def get_artist_link(link):
    '''
    For album page extension, navigates to desired album page, gets and parses HTML of webpage,
    scrapes and returns link for individual artist page
    '''
    # Getting and parsing HTML of album page
    base_url = 'https://www.albumoftheyear.org'
    url = base_url + link
    response = requests.get(url)
    page = response.content
    soup = BeautifulSoup(page, 'lxml')
    
    # Finding link for artist page
    artist_link = soup.find('div', class_='artist').find('a').get('href')
    return artist_link


# Functions for scraping artist pages
def scrape_artist(link):
    '''
    For an extension, navigates to desired webpage for an artist, gets and parses HTML of webpage,
    scrapes specified information/features, and returns features in dictionary

    Parameters
    ----------
    link: extension for webpage

    Returns
    ----------
    artist_dict: dictionary containing scraped information about artist
    '''
    # Getting and parsing HTML
    base_url = 'https://www.albumoftheyear.org'
    url = base_url + link
    response = requests.get(url)
    page = response.content
    soup_artist = BeautifulSoup(page, 'lxml')
    
    # Scraping features related to artist
    artist_name = soup_artist.find('h1', class_='artistHeadline').text.strip()
    artist_critic_score = get_artist_critic_score(soup_artist)
    artist_user_score = soup_artist.find('div', class_='artistUserScore').text.strip()
    num_user_ratings = get_artist_user_ratings(soup_artist)
    years_active = get_years_active(soup_artist)
    first_album = get_first_album(soup_artist)
    
    # Create dictionary
    artist_dict = {
        'artist': artist_name,
        'artist_critic_score': artist_critic_score,
        'artist_user_score': artist_user_score,
        'num_user_ratings': num_user_ratings,
        'years_active': years_active,
        'first_album': first_album
    }
    
    return artist_dict


# Helper functions for scraping artist page
def get_years_active(soup_artist):
    '''
    Returns:
    - how many years artist has been active based on discography if artist has released an album
    in at least two different years as datetime object
    - the year the album was released if artist has only one album listed as datetime object
    - NaN if no albums are listed
    '''
    artist_years = []
    
    for date in soup_artist.find('div', class_='facetContent').find_all('div', class_='date'):
        year = date.text.strip()
        if year == '0000':
            continue
        else:
            year = dt.datetime.strptime(year, '%Y')
            artist_years.append(year.year)
    
    if len(artist_years) > 1:
        artist_years.sort(reverse=True)
        years_active = artist_years[0] - artist_years[-1]
        return years_active
    elif len(artist_years) == 1:
        return year.year
    else:
        return np.nan


def get_artist_user_ratings(soup_artist):
    '''
    Returns number of user ratings for artist as integer
    '''
    try:
        user_ratings = soup_artist.find('div', class_="artistUserScoreBox").find('div', class_='text').strong.text.strip()
    except:
        return np.nan
    
    user_ratings = user_ratings.replace(',', '')
    return int(user_ratings)


def get_artist_critic_score(soup_artist):
    '''
    Returns critic score for atist as integer
    '''
    try:
        critic_score = soup_artist.find('div', class_='artistCriticScore').find('span').text.strip()
    except:
        return np.nan
    
    return int(critic_score)


def get_first_album(soup_artist):
    '''
    Returns name of first listed album of artist as cleaned string
    '''
    lp_parents = []
    lps = []

    types = soup_artist.find_all('div', class_ = 'type')
    for t in types:
        if re.search('LP', t.text):
            lp_parents.append(t.parent)

    for parent in lp_parents:
        if parent.find('div', class_='date').text != '0000':
            lps.append(parent.find('div', class_='albumTitle').text.strip())
    
    first_album = None

    if len(lps) > 1:
        first_album = lps[-1]
    else:
        for item in lps:
            first_album = item

    return first_album