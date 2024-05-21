# Discovering Leitmotifs in Multidimensional Time Series

This page was built in support of our paper "Discovering Leitmotifs in Multidimensional 
Time Series" by Patrick Schäfer and Ulf Leser.

Supporting Material
- `tests`: Please see the python tests for use cases
- `notebooks`: Please see the Jupyter Notebooks for use cases
- `csvs`: The results of the scalability experiments
- `leitmotifs`: Code implementing multidimensonal leitmotif discovery using LAMA
- `datasets`: Use cases in the paper

# Leitmotif Use Case

<img src="images/leitmotifs.png" width="500">

A **leitmotif** (*leading motif*) is a recurring theme or motif that carries 
symbolic significance in various forms of art, particularly literature, movies, 
and music. The distinct feature of any leitmotif is that humans associate them to 
meaning, which enhances narrative cohesion and establishes emotional connections 
with the audience. The use of (leit)motifs thus eases perception, interpretation, 
and identification with the underlying narrative. 
A genre that often uses leitmotifs are soundtracks, for instance in the compositions of 
Hans Zimmer or Howard Shore. The above figure shows a suite from *The Shire* with 14 
channels arranged by Howard Shore for Lord of the Rings. The suite opens and ends with 
the Hobbits' leitmotif, which is played by a solo tin whistle, and manifests in a 
distinct pattern in several, but not all channels of the piece.

Our LAMA (in purple) is the only method to correctly identify **4** 
occurrences within the leitmotif using a distinctive subset of channels. 
Other than EMD*, LAMA's occurrences show high pairwise similarity, too.

# Installation

The easiest is to use pip to install leitmotif.

## a) Install using pip
```
pip install leitmotif
```

You can also install the project from source.

## b) Build from Source

First, download the repository.
```
git clone https://github.com/patrickzib/leitmotif.git
```

Change into the directory and build the package from source.
```
pip install .
```

# Usage

The parameters of LAMA are:

- *n_dims* : Number of subdimensions to use
- *k_max* : The largest expected number of repeats. LAMA will search from  to  for motif sets
- *motif_length_range*

LAMA has a simple OO-API.

    ml = LAMA(
        ds_name,     # Name of the dataset
        series,      # Multidimensional time series
        n_dims,      # Number of sub-dimensions to use
        n_jobs,      # number of parallel jobs
    )
  
LAMA has a unique feature to automatically find suitable values for the motif length  and set size  so, that meaningful Leitmotifs of an input TS can be found without domain knowledge. The methods for determining values for  and  are based on an analysis of the extent function for different .

## Learning the leitmotif length 

To learn the motif length, we may simply call:

    ml.fit_motif_length(
        k_max,               # expected number of repeats
        motif_length_range,  # motif length range
        plot,                # Plot the results
        plot_elbows,         # Create an elbow plot 
        plot_motifsets,      # Plot the found motif sets
        plot_best_only       # Plot only the motif sets of the optimal length. Otherwise plot all local optima in lengths
    )    
To do variable length motif discovery simply set plot_best_only=False

## Learning the number of repeats

To do an elbow plot, and learn the number of repeats of the motif, we may simply call:

    ml.fit_k_elbow(
        k_max,                # expected number of repeats
        motif_length,         # motif length to use
        plot_elbows,          # Plot the elbow plot
        plot_motifsets        # Plot the found motif sets
    )
    
# Use Cases

Data Sets: We collected challenging real-life data sets to assess the quality and 
scalability of the algorithm. An overview of datasets can be found in Table 2 
of our paper. 

- Jupyter-Notebooks for finding subdimensional Leitmotif in multidimensional time series
<a href="notebooks/use_cases_paper.ipynb">Multivariate Use Case</a>:
highlights a use case used in the paper, and shows the unique ability 
to learn its parameters from the data and find interesting motif sets.

- All other use cases can be found in the <a href="tests">test folder</a>


## Citation
If you use this work, please cite as:

TODO
