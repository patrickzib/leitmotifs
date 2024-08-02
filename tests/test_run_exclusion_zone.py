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

method_names = [
    "LAMA (alpha=0)",
    "LAMA (alpha=0.25)",
    "LAMA (alpha=0.5)",
    "LAMA (alpha=0.75)",
    "LAMA (alpha=1)"
]

all_plot_names = {
    "_exclusion": [
        "LAMA (alpha=0)",
        "LAMA (alpha=0.25)",
        "LAMA (alpha=0.5)",
        "LAMA (alpha=0.75)",
        "LAMA (alpha=1)"
    ]
}


# Run all tests
audio.test_publication(method_names=method_names)
crypto.test_publication(method_names=method_names)
motion.test_publication(method_names=method_names)
physiodata.test_publication(method_names=method_names)
birds.test_publication(method_names=method_names)
soundtracks.test_publication(method_names=method_names)

# Evaluate all tests
audio.test_plot_results(
    plot=False, method_names=method_names, all_plot_names=all_plot_names)
crypto.test_plot_results(
    plot=False, method_names=method_names, all_plot_names=all_plot_names)
motion.test_plot_results(
    plot=False, method_names=method_names, all_plot_names=all_plot_names)
physiodata.test_plot_results(
    plot=False, method_names=method_names, all_plot_names=all_plot_names)
birds.test_plot_results(
    plot=False, method_names=method_names, all_plot_names=all_plot_names)
soundtracks.test_plot_results(
    plot=False, method_names=method_names, all_plot_names=all_plot_names)