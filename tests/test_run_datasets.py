import sys
sys.path.append('..')
sys.path.append('../leitmotifs')

from joblib import Parallel, delayed

import multivariate_audio_test as audio
import multivariate_birdsounds_test as birds
import multivariate_crypto_test as crypto
import multivariate_motion_test as motion
import multivariate_physiodata_test as physiodata
import multivariate_soundtracks_test as soundtracks

print("Running all tests")


# Define the functions to run tests
def run_test(test_function):
    test_function(plot=False)


# Define the functions to evaluate tests
def evaluate_test(test_function):
    test_function(plot=False)

server = False

# Parallel for server only?
if server:
    # List of test functions
    test_functions = [
        audio.test_publication,
        crypto.test_publication,
        motion.test_publication,
        physiodata.test_publication,
        birds.test_publication,
        soundtracks.test_publication
    ]

    # List of evaluation functions
    evaluation_functions = [
        audio.test_plot_results,
        crypto.test_plot_results,
        motion.test_plot_results,
        physiodata.test_plot_results,
        birds.test_plot_results,
        soundtracks.test_plot_results
    ]

    # Run all tests in parallel
    Parallel(n_jobs=-1)(delayed(run_test)(func) for func in test_functions)

    # Evaluate all tests in parallel
    Parallel(n_jobs=-1)(delayed(evaluate_test)(func) for func in evaluation_functions)
else:

    # Run all tests
    audio.test_publication()
    crypto.test_publication()
    motion.test_publication()
    physiodata.test_publication()
    birds.test_publication()
    soundtracks.test_publication()

    # Evaluate all tests
    audio.test_plot_results(plot=False)
    crypto.test_plot_results(plot=False)
    motion.test_plot_results(plot=False)
    physiodata.test_plot_results(plot=False)
    birds.test_plot_results(plot=False)
    soundtracks.test_plot_results(plot=False)
