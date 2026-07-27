import _repo_path  # noqa: F401

from joblib import Parallel, delayed

print("Running all dataset benchmarks")


# Define the functions to run tests
def run_test(test_function):
    test_function(plot=False)


# Define the functions to evaluate tests
def evaluate_test(test_function):
    test_function(plot=False)

def main(server=False):
    import run_audio_benchmark as audio
    import run_birdsounds_benchmark as birds
    import run_crypto_benchmark as crypto
    import run_motion_benchmark as motion
    import run_physiodata_benchmark as physiodata
    import run_soundtracks_benchmark as soundtracks

    test_functions = [
        audio.test_publication,
        crypto.test_publication,
        motion.test_publication,
        physiodata.test_publication,
        birds.test_publication,
        soundtracks.test_publication
    ]

    evaluation_functions = [
        audio.test_plot_results,
        crypto.test_plot_results,
        motion.test_plot_results,
        physiodata.test_plot_results,
        birds.test_plot_results,
        soundtracks.test_plot_results
    ]

    if server:
        Parallel(n_jobs=-1)(delayed(run_test)(func) for func in test_functions)
        Parallel(n_jobs=-1)(
            delayed(evaluate_test)(func) for func in evaluation_functions)
    else:
        for func in test_functions:
            func()

        for func in evaluation_functions:
            func(plot=False)


if __name__ == "__main__":
    main()
