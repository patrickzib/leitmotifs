# Discovering Leitmotifs (Subdimensional Motif Sets) in Multidimensional Time Series

This page was built in support of our paper "Discovering Leitmotifs in Multidimensional 
Time Series" by Patrick Schäfer and Ulf Leser.

A leitmotif is a recurring theme in literature, movies or music that carries symbolic significance for the piece it is contained in. When this piece can be represented as a multi-dimensional time series (MDTS), such as acoustic or visual observations, finding a leitmotif is equivalent to the pattern discovery problem, which is an unsupervised and complex problem in time series analytics. Compared to the univariate case, it carries additional complexity because patterns typically do not occur in all dimensions but only in a few - which are, however, unknown and must be detected by the method itself. In this paper, we present the novel, efficient and highly effective leitmotif discovery algorithm **LAMA** for MDTS. **LAMA** rests on two core principals: (a) a leitmotif manifests solely given a yet unknown number of sub-dimensions - neither too few, nor too many, and (b) the set of sub-dimensions are not independent from the best pattern found therein, necessitating both problems to be approached in a joint manner. In contrast to most previous methods, **LAMA** tackles both problems jointly - instead of first selecting dimensions (or leitmotifs) and then finding the best leitmotifs (or dimensions). 

Supporting Material
- `tests`: Please see the python tests for use cases
- `notebooks`: Please see the Jupyter Notebooks for use cases
- `csvs`: The results of the scalability experiments
- `leitmotifs`: Code implementing multidimensonal leitmotif discovery using LAMA
- `datasets`: Use cases in the paper

# Mining Leitmotifs - Use Case

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

You may listen to the song on youtube - starting from second ~6, the same melody (leitmotif) is played by the tin whistle twice, and the this theme is repeated after 2 minutes into the song: 

[![The Shire](https://raw.githubusercontent.com/patrickzib/leitmotifs/main/images/hobbit_theme.jpg)](http://www.youtube.com/watch?v=SRgBI0WqWp4&t=7s "Video Title")

The result of leitmotif discovery is shown next:

<img src="https://raw.githubusercontent.com/patrickzib/leitmotifs/main/images/leitmotifs.png" width="100%">

Our **LAMA (in brown)** is the only method to correctly identify **4** 
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

# Usage of LAMA

The three hyper-parameters of **LAMA** are:
- *n_dims* : Number of subdimensions to use
- *k_max* : The largest expected number of repeats. LAMA will search from  to  for motif sets
- *motif_length_range*: The range of lengths to test

LAMA has a simple OO-API.

```python
    from leitmotifs.plotting import *
    ml = LAMA(
        ds_name,     # Name of the dataset
        series,      # Multidimensional time series
        distance,    # Distance measure used, default: z-normed ED
        n_dims,      # Number of sub-dimensions to use
        n_jobs,      # number of parallel jobs
    )
```

The result will look like this, indicating the found motif set to the right, its dimensions and the k locations to the bottom.
<img src="https://raw.githubusercontent.com/patrickzib/leitmotifs/main/images/lotr-motifset-lama.png" width="100%">

LAMA has a unique feature to automatically find suitable values for the motif length  and set size  so, that meaningful Leitmotifs of an input TS can be found without domain knowledge. The methods for determining values for $k$ and $l$ are based on an analysis of the extent function for different input value ranges.

## Learning the Leitmotif length 

To learn the motif length, we may simply call:

```python
    ml.fit_motif_length(
        k_max,               # expected number of repeats
        motif_length_range,  # motif length range
        plot,                # Plot the results
        plot_elbows,         # Create an elbow plot 
        plot_motifsets,      # Plot the found motif sets
        plot_best_only       # Plot only the motif sets of the optimal length. Otherwise plot all local optima in lengths
    )
```
    
To do variable length motif discovery simply set `plot_best_only=False`

The generated plots looks like this, with good window lengths at local minima:

<img src="https://raw.githubusercontent.com/patrickzib/leitmotifs/main/images/window_length_selection.png" width="600">


## Learning the number of repeats

To do an elbow plot, and learn the number of repeats of the motif, we may simply call:

```python
    ml.fit_k_elbow(
        k_max,                # expected number of repeats
        motif_length,         # motif length to use
        plot_elbows,          # Plot the elbow plot
        plot_motifsets        # Plot the found motif sets
    )
```
The generated plots looks like this, with good number of repeats at local minima:

<img src="https://raw.githubusercontent.com/patrickzib/leitmotifs/main/images/elbow_points.png" width="600">
    
# Use Cases

Data Sets: We collected and annotated 14 challenging real-life data sets to assess the quality and 
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
        <tr>
          <td>Physiodata (Physical Exercises)</td>
          <td>Wearable Sensors</td>
          <td>5526</td>
          <td>5</td>
          <td>20</td>
        </tr>
        <tr>
          <td>Bitcoin Halving</td>
          <td>Crypto/Stocks</td>
          <td>3591</td>
          <td>5</td>
          <td>3</td>
        </tr>          
      </tbody>
    </table>
  </small>
</table>

## Aggregated Results


| Method              |   Precision, mean |   Precision, median |   Recall, mean |   Recall, median |
|:--------------------|------------------------:|--------------------------:|---------------------:|-----------------------:|
| EMD*                |                 59.3 |                   65      |              75.9 |                     80 |
| K-Motifs (TOP-f)    |                 61.1 |                   70      |              70.8 |                    **100** |
| K-Motifs (all dims) |                 76.8 |                   83.3    |              82.6 |                    **100** |
| SMM                 |                 31.8 |                   26.5    |              65.4 |                     95 |
| mSTAMP              |                 53.8 |                  **100**      |              36.7 |                     20 |
| mSTAMP+MDL          |                 46.2 |                    0      |              29.0 |                      0 |
| LAMA                |                 **88.7** |                  **100**      |              **95.1** |                    **100** |

See all results in <a href="notebooks/plot_ground_truth.ipynb">Results Notebook</a>.

## Notebooks

- Jupyter-Notebooks for finding subdimensional Leitmotifs in a multidimensional time series
<a href="notebooks/use_case.ipynb">Multivariate Use Case</a>:
highlights a use case used in the paper, and shows the unique ability 
to learn its parameters from the data and find interesting motif sets.

- Jupter-Notebook showcasing SMM-Results (SMM ran using Matlab): <a href="notebooks/smm_motif_plot.ipynb">SMM-Results</a>.

- Jupter-Notebook showcasing using Leitmotifs for Summarization: <a href="notebooks/har_pamap.ipynb">Summarization</a>.

- Jupter-Notebook showcasing BitCoint-Halving Events: <a href="notebooks/crypto.ipynb">Bitcoin-Halving</a>.

- All other use cases presented in the paper can be found in the <a href="tests">test folder</a>


## Citation
If you use this work, please cite as:
```
@article{leitmotifs2025,
  title={Discovering Leitmotifs in Multidimensional Time Series},
  author={Schäfer, Patrick and Leser, Ulf},
  journal={Proceedings of the VLDB Endowment},
  volume={18},
  number={2},
  pages={377-389},
  year={2025},
  publisher={PVLDB}
}
```

Link to the <a href="https://www.vldb.org/pvldb/vol18/p377-schafer.pdf">paper</a>.
