import multivariate_audio_test as audio
import multivariate_birdsounds_test as birds
import multivariate_crypto_test as crypto
import multivariate_motion_test as motion
import multivariate_physiodata_test as physiodata
import multivariate_soundtracks_test as soundtracks

dataset_names = []

i = 1

df, df_gt = physiodata.load_physiodata()
print("Writing Physio-Data", df.T.shape)
# df.T.to_csv("smm_benchmark/" + str(i) + "_" + "physio.csv", index=False, header=None)
df.T.to_csv("smm_benchmark/" + str(i) + ".csv", index=False, header=None)
dataset_names.append("physio")

for dataset_name in [
    "Boxing",
    "Swordplay",
    "Basketball",
    "Charleston - Side By Side Female"]:
    i = i+1
    motion.get_ds_parameters(dataset_name)
    df, df_gt, _, _ = motion.read_motion_dataset()
    print("Writing" + dataset_name, df.T.shape)
    # df.T.to_csv("smm_benchmark/" + str(i) + "_" + dataset_name + ".csv", index=False,
    #            header=None)
    df.T.to_csv("smm_benchmark/" + str(i) + ".csv", index=False, header=None)
    dataset_names.append(dataset_name)

i = i+1
df, df_gt = crypto.load_crypto()
print("Writing Crypto-Data", df.T.shape)
# df.T.to_csv("smm_benchmark/" + str(i) + "_" + "crypto.csv", index=False, header=None)
df.T.to_csv("smm_benchmark/" + str(i) + ".csv", index=False, header=None)
dataset_names.append("crypto")

i = i+1
birds.get_ds_parameters("Common-Starling")
_, df, _, _ = birds.read_audio_from_dataframe(birds.pandas_file_url, birds.channels)
print("Writing Starling-Data", df.T.shape)
# df.T.to_csv("smm_benchmark/" + str(i) + "_" + "birds.csv", index=False, header=None)
df.T.to_csv("smm_benchmark/" + str(i) + ".csv", index=False, header=None)
dataset_names.append("birds")

for dataset_name in [
    "What I've Done - Linkin Park",
    "Numb - Linkin Park",
    "Vanilla Ice - Ice Ice Baby",
    "Queen David Bowie - Under Pressure",
    "The Rolling Stones - Paint It, Black"]:
    i = i+1
    audio.get_ds_parameters(dataset_name)
    _, df, _, _ = audio.read_audio_from_dataframe(audio.pandas_file_url, audio.channels)
    print("Writing " + dataset_name, df.T.shape)
    # df.T.to_csv("smm_benchmark/" + str(i) + "_" + dataset_name + ".csv", index=False,
    #             header=None)
    df.T.to_csv("smm_benchmark/" + str(i) + ".csv", index=False, header=None)
    dataset_names.append(dataset_name)

for dataset_name in [
    "Star Wars - The Imperial March",
    "Lord of the Rings Symphony - The Shire"
]:
    i = i+1
    soundtracks.get_ds_parameters(dataset_name)
    _, df, _, _ = soundtracks.read_audio_from_dataframe(
        soundtracks.pandas_file_url, soundtracks.channels)
    print("Writing " + dataset_name, df.T.shape)
    # df.T.to_csv("smm_benchmark/" + str(i) + "_" + dataset_name + ".csv", index=False,
    #             header=None)
    df.T.to_csv("smm_benchmark/" + str(i) + ".csv", index=False, header=None)
    dataset_names.append(dataset_name)

print(dataset_names)