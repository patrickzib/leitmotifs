import os
import wave
import librosa
from os.path import exists

import motiflets.motiflets as ml
from motiflets.competitors import *
from motiflets.plotting import *

import subprocess
import matplotlib
from pydub import AudioSegment
import pylrc

def plot_motiflet(series, motiflet, motif_length, title=None):
    # turn into 2d array
    if series.ndim != 2:
        raise ValueError('The input dimension must be 2d.')

    fig, axes = plt.subplots(series.shape[0], 1, figsize=(6, 2 + series.shape[0]))

    # one plot by dimension
    for dim in range(0, series.shape[0]):
        if isinstance(series, np.ndarray):
            values = pd.DataFrame(data=np.array(
                [series[dim, i:i+motif_length] for i in motiflet]))
        else:
            values = pd.DataFrame(data=np.array(
                [series.iloc[dim, i:i+motif_length] for i in motiflet]))
            axes[dim].set_title(series.index[dim], fontsize=8)

        ax = sns.lineplot(data=values.melt().set_index("variable"),
                          ci=95, n_boot=10,
                          ax = axes[dim])

        ax.legend().set_visible(False)
        ax.axis('off')

    if title:
        plt.suptitle(title)

    fig.patch.set_visible(False)
    fig.tight_layout()
    sns.despine()
    return fig, ax


def read_audio(file, mono):
    amplitude, channels = librosa.load(file, mono=mono)    
    mfcc_feat = librosa.feature.mfcc(y=amplitude, sr=channels)
    audio_length_d = librosa.get_duration(filename=file)
    return amplitude, channels, mfcc_feat, audio_length_d


def read_lrc(url, offset_min=0, offset_sec=0, offset_msec=0, debug=False):
    file_d = open(str(url))
    lrc_string_d = ''.join(file_d.readlines())
    file_d.close()
    
    subs = pylrc.parse(lrc_string_d)
    if offset_min > 0 or offset_sec > 0 or offset_msec > 0:
        for sub in subs:
            sub.shift(
                offset_min, 
                offset_sec,
                  offset_msec)  # offset by 00:00.000 (difference of less than a second in between Youtube video (https://www.youtube.com/watch?v=nbwn21TU4ec) & lrc file was manually estimated and verified)
        if debug:
            print(subs.__dict__)
    for sub in subs:
        if debug:
            print(str(sub.seconds + sub.minutes * 60) + ":" + " " + sub.text)
    return subs

def get_dataframe_from_subtitle_object(subtitle):
    tmp_df_d = pd.DataFrame()
    subtitles_array_d = []
    subtitles_timestamp_d = []
    subtitle_duration_d = []

    for sub in subtitle:
        subtitles_array_d.append(sub.text)
        subtitles_timestamp_d.append(sub.time)
    counter_d = 0
    for timestamp in subtitles_timestamp_d:
        timestamp_list_length = len(subtitles_timestamp_d) - 1

        if timestamp_list_length == counter_d:
            subtitle_duration_d.append(0)
        else:
            subtitle_duration_d.append(subtitles_timestamp_d[counter_d + 1] - subtitles_timestamp_d[counter_d])
        counter_d = counter_d + 1

        tmp_df_d['subtitles'] = subtitles_array_d
    tmp_df_d['seconds'] = subtitles_timestamp_d
    tmp_df_d['duration'] = subtitle_duration_d
    return tmp_df_d

def lookup_lyrics(df, position, motif_length):
    text = ""
    for i in df.index:
        if (i <= position <= i + df.loc[i, "duration"]//2 )or  \
           (i <= position + motif_length <= i + df.loc[i, "duration"]):
            if df.loc[i, "subtitles"].strip():
                if text:
                    text += "\n"
                text += " " + df.loc[i, "subtitles"]
    return text