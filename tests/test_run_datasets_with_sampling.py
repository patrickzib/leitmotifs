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

def test():
    for sampling_factor in [2, 3, 4, 5, 6, 7, 8]:
        print ("Running tests with sampling", sampling_factor)
        crypto.sampling_factor = sampling_factor
        motion.sampling_factor = sampling_factor
        physiodata.sampling_factor = sampling_factor
        lama.sampling_factor = sampling_factor

        # Run all tests
        # FIXME: window lengths are not adapted to the sampling factor

        #audio.test_publication(sampling_factor=sampling_factor)
        #crypto.test_publication()
        motion.test_publication()
        #physiodata.test_publication()
        #birds.test_publication(sampling_factor=sampling_factor)
        #soundtracks.test_publication(sampling_factor=sampling_factor)

        # Evaluate all tests
        #audio.test_plot_results(plot=False, sampling_factor=sampling_factor)
        #crypto.test_plot_results(plot=False)
        motion.test_plot_results(plot=False)
        #physiodata.test_plot_results(plot=False)
        #birds.test_plot_results(plot=False, sampling_factor=sampling_factor)
        #soundtracks.test_plot_results(plot=False, sampling_factor=sampling_factor)