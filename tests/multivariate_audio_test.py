import matplotlib as mpl

from audio.lyrics import *

mpl.rcParams['figure.dpi'] = 300

path = "../../motiflets_use_cases/audio/"

datasets = {
    "Numb - Linkin Park": {
        "ks": 10,
        "channels": 3,
        "length_in_seconds": 4,
        "ds_name": "Numb - Linkin Park",
        "audio_file_url": path + "Numb - Linkin Park.wav",
        "lrc_url": path + "Numb - Linkin Park.lrc"
    },
    "What I've Done - Linkin Park": {
        "ks": 10,
        "channels": 3,
        "length_in_seconds": 7.75,
        "ds_name": "What I've Done - Linkin Park",
        "audio_file_url": path + "What I've Done - Linkin Park.wav",
        "lrc_url": path + "What I've Done - Linkin Park.lrc"
    },
    "The Rolling Stones - Paint It, Black": {
        "ks": 15,
        "channels": 3,
        "length_in_seconds": 5.4,
        "ds_name": "The Rolling Stones - Paint It, Black",
        "audio_file_url": path + "The Rolling Stones - Paint It, Black.wav",
        "lrc_url": path + "The Rolling Stones - Paint It, Black.lrc"
    }
}

dataset = datasets["The Rolling Stones - Paint It, Black"]
k_max = dataset["ks"]
channels = dataset["channels"]
length_in_seconds = dataset["length_in_seconds"]
ds_name = dataset["ds_name"]
audio_file_url = dataset["audio_file_url"]
lrc_url = dataset["lrc_url"]


def read_songs():
    # Read audio from wav file
    x, sr, mfcc_f, audio_length_seconds = read_audio(audio_file_url, True)
    print("Length:", audio_length_seconds)
    index_range = np.arange(0, mfcc_f.shape[1]) * audio_length_seconds / mfcc_f.shape[1]
    df = pd.DataFrame(mfcc_f,
                      index=["MFCC " + str(a) for a in np.arange(0, mfcc_f.shape[0])],
                      columns=index_range
                      )
    df.index.name = "MFCC"
    return audio_length_seconds, df, index_range


def test_dendrogram():
    channels = 10
    audio_length_seconds, df, index_range = read_songs()
    df = df.iloc[0:channels]

    motif_length = int(length_in_seconds / audio_length_seconds * df.shape[1])
    print(motif_length)

    ml = Motiflets(ds_name, df,
                   elbow_deviation=1.25,
                   slack=1.0,
                   dimension_labels=df.index
                   )

    ml.fit_dendrogram(k_max, motif_length, n_clusters=3)


def test_dendrogram_2():
    channels = 10
    audio_length_seconds, df, index_range = read_songs()
    df = df.iloc[0:channels]

    motif_length = int(length_in_seconds / audio_length_seconds * df.shape[1])
    print(motif_length)

    ml = Motiflets(ds_name, df,
                   # elbow_deviation=1.25,
                   # slack=1.0,
                   dimension_labels=df.index
                   )

    ml.fit_dendrogram(k_max, motif_length, n_clusters=2)


def test_audio():
    audio_length_seconds, df, index_range = read_songs()

    # df = df.iloc[:channels]
    # channels = ['MFCC 0', 'MFCC 1', 'MFCC 2']
    # channels = ['MFCC 0', 'MFCC 1', 'MFCC 2', 'MFCC 3']
    # channels = ['MFCC 2', 'MFCC 3']  ## hmm
    # channels = ['MFCC 4', 'MFCC 5']
    # channels = ['MFCC 0', 'MFCC 1', 'MFCC 5']
    # channels = ['MFCC 1', 'MFCC 5', 'MFCC 4']
    # channels = ['MFCC 0', 'MFCC 1', 'MFCC 2', 'MFCC 3', 'MFCC 4',
    #             'MFCC 5', 'MFCC 6', 'MFCC 7', 'MFCC 8', 'MFCC 9']

    channels = ['MFCC 1', 'MFCC 2' , 'MFCC 3', 'MFCC 7']  # 2 Motifs, hmm
    # channels = ['MFCC 4', 'MFCC 8', 'MFCC 6', 'MFCC 9']    # hmmm

    df = df.loc[channels]

    motif_length = int(length_in_seconds / audio_length_seconds * df.shape[1])
    print(motif_length, length_in_seconds, "s")

    subtitles = read_lrc(lrc_url)
    df_sub = get_dataframe_from_subtitle_object(subtitles)
    df_sub.set_index("seconds", inplace=True)

    motif_length_range_in_s = np.arange(4, 5.8, 0.1)
    motif_length_range = np.int32(motif_length_range_in_s /
                                  audio_length_seconds * df.shape[1])

    ml = Motiflets(ds_name, df,
                   #elbow_deviation=1.25,
                   #slack=1.0,
                   dimension_labels=df.index
                   )
    motif_length, _ = ml.fit_motif_length(
        k_max, motif_length_range, subsample=1)

    #    motif_length = motif_length_range[m]
    #    motif_length_in_seconds = motif_length_range_in_s[m]
    dists, motiflets, elbow_points = ml.fit_k_elbow(
        k_max,
        motif_length=motif_length,
        plot_elbows=False,
        plot_motifs_as_grid=False
    )

    # best motiflet
    motiflet = np.sort(motiflets[elbow_points[-1]])
    print("Positions:", index_range[motiflet])

    lyrics = []
    for i, m in enumerate(motiflet):
        l = lookup_lyrics(df_sub, index_range[m], length_in_seconds)
        lyrics.append(l)
        print(i + 1, l)

    path_ = "audio/snippets/" + ds_name + "_Channels_" + str(
        len(df.index)) + "_Motif.pdf"
    ml.plot_motifset(path_)

    extract_audio_segment(
        df, ds_name, audio_file_url, "snippets",
        length_in_seconds, index_range, motif_length, motiflet)



def test_audio_window_length():
    audio_length_seconds, df, index_range = read_songs()

    # df = df.iloc[:channels]
    channels = ['MFCC 0', 'MFCC 1', 'MFCC 2']
    df = df.loc[channels]

    motif_length = int(length_in_seconds / audio_length_seconds * df.shape[1])
    print(motif_length, length_in_seconds, "s")

    subtitles = read_lrc(lrc_url)
    df_sub = get_dataframe_from_subtitle_object(subtitles)
    df_sub.set_index("seconds", inplace=True)

    motif_length_range_in_s = np.arange(4, 5.8, 0.1)
    motif_length_range = np.int32(motif_length_range_in_s /
                                  audio_length_seconds * df.shape[1])

    ml = Motiflets(ds_name, df,
                   elbow_deviation=1.25,
                   slack=1.0,
                   dimension_labels=df.index
                   )

    best_length, _ = ml.fit_motif_length(
            k_max, motif_length_range,
            subsample=1,
            plot_elbows=True,
            plot_motifs_as_grid=False
            )

    ml.plot_motifset()
    print("Best found length", best_length)

    exclusion = ml.motiflets[ml.elbow_points]
    best_length, all_extrema = ml.fit_motif_length(
        k_max,
        motif_length_range,
        subsample=1,
        plot_elbows=True,
        plot_motifs_as_grid=False,
        exclusion=exclusion,
        exclusion_length=best_length
    )

    print("Best found length", best_length)

    motiflet = np.sort(ml.motiflets[ml.elbow_points[-1]])

    lyrics = []
    for i, m in enumerate(motiflet):
        l = lookup_lyrics(df_sub, index_range[m], length_in_seconds)
        lyrics.append(l)
        print(i + 1, l)

    ml.plot_motifset()

    extract_audio_segment(
        df, ds_name, audio_file_url, "snippets",
        length_in_seconds, index_range, motif_length, motiflet)