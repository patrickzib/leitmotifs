# Discovering Leitmotifs in Multidimensional Time Series

This page was built in support of our paper "Discovering Leitmotifs in Multidimensional 
Time Series" by Patrick Schäfer and Ulf Leser.

A leitmotif is a recurring theme in literature, movies or music that carries symbolic significance for the piece it is contained in. When this piece can be represented as a multi-dimensional time series (MDTS), such as acoustic or visual observations, finding a leitmotif is equivalent to the pattern discovery problem, which is an unsupervised and complex problem in time series analytics. Compared to the univariate case, it carries additional complexity because patterns typically do not occur in all dimensions but only in a few - which are, however, unknown and must be detected by the method itself. In this paper, we present the novel, efficient and highly effective leitmotif discovery algorithm LAMA for MDTS. LAMA rests on two core principals: (a) a leitmotif manifests solely given a yet unknown number of sub-dimensions - neither too few, nor too many, and (b) the set of sub-dimensions are not independent form the best pattern found therein, necessitating both problems to be approached in a joint manner. In contrast to all previous methods, LAMA is the first to tackle both problems jointly - instead of first selecting dimensions (or leitmotifs) and then finding the best leitmotifs (or dimensions). 

Supporting Material
- `tests`: Please see the python tests for use cases
- `notebooks`: Please see the Jupyter Notebooks for use cases
- `csvs`: The results of the scalability experiments
- `leitmotifs`: Code implementing multidimensonal leitmotif discovery using LAMA
- `datasets`: Use cases in the paper

# Leitmotif Use Case

<img src="https://raw.githubusercontent.com/patrickzib/leitmotifs/main/images/leitmotifs.png" width="500">

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
git clone https://github.com/patrickzib/leitmotifs.git
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

## Learning the Leitmotif length 

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

Data Sets: We collected and annotated 12 challenging real-life data sets to assess the quality and 
scalability of the LAMA algorithm. 

<table>
  <caption>Ground leitmotifs were manually inferred. GT refers to the number of leitmotif occurrences.</caption>
  <small>
    <table>
      <thead>
        <tr>
          <th>Use Case</th>
          <th>Category</th>
          <th>Length</th>
          <th>Dim.</th>
          <th>GT</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Charleston</td>
          <td>Motion Capture</td>
          <td>506</td>
          <td>93</td>
          <td>3</td>
        </tr>
        <tr>
          <td>Boxing</td>
          <td>Motion Capture</td>
          <td>4840</td>
          <td>93</td>
          <td>10</td>
        </tr>
        <tr>
          <td>Swordplay</td>
          <td>Motion Capture</td>
          <td>2251</td>
          <td>93</td>
          <td>6</td>
        </tr>
        <tr>
          <td>Basketball</td>
          <td>Motion Capture</td>
          <td>721</td>
          <td>93</td>
          <td>5</td>
        </tr>
        <tr>
          <td>LOTR - The Shire</td>
          <td>Soundtrack</td>
          <td>6487</td>
          <td>20</td>
          <td>4</td>
        </tr>
        <tr>
          <td>SW - The Imperial March</td>
          <td>Soundtrack</td>
          <td>8015</td>
          <td>20</td>
          <td>5</td>
        </tr>
        <tr>
          <td>RS - Paint it black</td>
          <td>Pop Music</td>
          <td>9744</td>
          <td>20</td>
          <td>10</td>
        </tr>
        <tr>
          <td>Linkin Park - Numb</td>
          <td>Pop Music</td>
          <td>8018</td>
          <td>20</td>
          <td>5</td>
        </tr>
        <tr>
          <td>Linkin P. - What I've Done</td>
          <td>Pop Music</td>
          <td>8932</td>
          <td>20</td>
          <td>6</td>
        </tr>
        <tr>
          <td>Queen - Under Pressure</td>
          <td>Pop Music</td>
          <td>9305</td>
          <td>20</td>
          <td>16</td>
        </tr>
        <tr>
          <td>Vanilla Ice - Ice Ice Baby</td>
          <td>Pop Music</td>
          <td>11693</td>
          <td>20</td>
          <td>20</td>
        </tr>
        <tr>
          <td>Starling</td>
          <td>Wildlife Rec.</td>
          <td>2839</td>
          <td>20</td>
          <td>4</td>
        </tr>
      </tbody>
    </table>
  </small>
</table>

## Aggregated Results


| Method              |   Mean Precision        |   Median Precision        |  Mean Recall         |   Median Recall        |
|:--------------------|------------------------:|--------------------------:|---------------------:|-----------------------:|
| EMD*                |                0.656548 |                      0.8  |             0.8      |                    0.8 |
| K-Motifs (TOP-f)    |                0.692857 |                      0.75 |             0.814286 |                    1   |
| K-Motifs (all dims) |                0.906548 |                      1    |             0.942857 |                    1   |
| LAMA                |                0.916071 |                      1    |             0.985714 |                    1   |
| mSTAMP              |                0.571429 |                      1    |             0.3      |                    0.2 |
| mSTAMP+MDL          |                0.571429 |                      1    |             0.3      |                    0.2 |

See all results in <a href="notebooks/plot_ground_truth.ipynb">Results Notebook</a>.

## Notebooks


- Jupyter-Notebooks for finding subdimensional Leitmotifs in a multidimensional time series
<a href="notebooks/use_case.ipynb">Multivariate Use Case</a>:
highlights a use case used in the paper, and shows the unique ability 
to learn its parameters from the data and find interesting motif sets.

- All other use cases presented in the paper can be found in the <a href="tests">test folder</a>


## Citation
If you use this work, please cite as:

TODO