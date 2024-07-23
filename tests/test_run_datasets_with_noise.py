import sys

sys.path.append('..')
sys.path.append('../leitmotifs')

import multivariate_audio_test as audio
import multivariate_birdsounds_test as birds
import multivariate_crypto_test as crypto
import multivariate_motion_test as motion
import multivariate_physiodata_test as physiodata
import multivariate_soundtracks_test as soundtracks
import leitmotifs.lama as lama

for noise_level in reversed([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]):
    print ("Running tests with noise level", noise_level)
    crypto.noise_level = noise_level
    motion.noise_level = noise_level
    physiodata.noise_level = noise_level
    lama.noise_level = noise_level

    # Run all tests
    soundtracks.test_publication(noise_level=noise_level)
    audio.test_publication(noise_level=noise_level)
    crypto.test_publication()
    motion.test_publication()
    physiodata.test_publication()
    birds.test_publication(noise_level=noise_level)

    # Evaluate all tests
    soundtracks.test_plot_results(plot=False, noise_level=noise_level)
    audio.test_plot_results(plot=False, noise_level=noise_level)
    crypto.test_plot_results(plot=False)
    motion.test_plot_results(plot=False)
    physiodata.test_plot_results(plot=False)
    birds.test_plot_results(plot=False, noise_level=noise_level)
