# Discovering Leitmotifs in Multidimensional Time Series

This page was built in support of our paper "Discovering Leitmotifs in Multidimensional 
Time Series" by Patrick Schäfer and Ulf Leser.

Supporting Material
- `tests`: Please see the python tests for use cases
- `notebooks`: Please see the Jupyter Notebooks for use cases
- `csvs`: The results of the scalability experiments
- `motiflets`: Code implementing multidimensonal k-Motiflet
- `datasets`: Use cases in the paper

# Showcase

TODO 

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

The parameters of motiflets are:

- *n_dims* : Number of subdimensions to use
- *k_max* : The largest expected number of repeats. Motiflets will search from  to  for motif sets
- *motif_length_range*

Motiflets have a simple OO-API.

    ml = Motiflets(
        ds_name,     # Name of the dataset
        series,      # Multidimensional time series
        n_dims,      # Number of subdimensions to use
        n_jobs,      # number of parallel jobs
    )
  
Motiflets have a unique feature to automatically find suitable values for the motif length  and set size  so, that meaningful Leitmotifs of an input TS can be found without domain knowledge. The methods for determining values for  and  are based on an analysis of the extent function for different .

## Learning the motif length 

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

- Jupyter-Notebook from the univariate paper
<a href="notebooks/use_cases_paper.ipynb">Univariate Use Cases</a>:
highlights all use cases used in the paper and shows the unique ability 
to learn its parameters from the data and find interesting motif sets.



## Citation
If you use this work, please cite as:

TODO
