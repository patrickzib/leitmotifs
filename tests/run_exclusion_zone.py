import _repo_path  # noqa: F401

from joblib import Parallel, delayed

print("Running exclusion-zone benchmarks")

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


def main():
    import run_audio_benchmark as audio
    import run_birdsounds_benchmark as birds
    import run_crypto_benchmark as crypto
    import run_motion_benchmark as motion
    import run_physiodata_benchmark as physiodata
    import run_soundtracks_benchmark as soundtracks

    audio.test_publication(method_names=method_names)
    crypto.test_publication(method_names=method_names)
    motion.test_publication(method_names=method_names)
    physiodata.test_publication(method_names=method_names)
    birds.test_publication(method_names=method_names)
    soundtracks.test_publication(method_names=method_names)

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


if __name__ == "__main__":
    main()
