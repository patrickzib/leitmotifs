from audio.lyrics import *
import urllib.request

def test_grap_stream():
    stream_url = "https://ia800904.us.archive.org/6/items/HarryPotterHedwigsTheme/Harry%20Potter%20-%20Hedwigs%20Theme.mp3"
    urllib.request.urlretrieve(stream_url, "audio/hedwig.mp3")
