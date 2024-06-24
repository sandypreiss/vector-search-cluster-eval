# Vector Search Cluster Evaluation

This repo tests a vector search approach to evaluating text clusters. Results presented at [JSM 2024](https://ww2.amstat.org/meetings/jsm/2024/).

The proposed approach to evaluate text clusters is:
1. Fit a clustering model to a corpus of texts
1. Generate names for the clusters using a LLM
1. For each cluster:
    1. Find the texts that are nearest neighbors to the cluster name
    1. Calculate precision and recall for the nearest neighbors against the texts in the cluster

The theory is that if the cluster is good (distinctive and cohesive) and the cluster name accurately represents the cluster, precision and recall should be high. If precision and recall or low, either the cluster or the cluster name (or both) is not good.

To test this approach empirically, we conduct an experiment using the [20 Newsgroups](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html) dataset. This dataset contains newsgroups posts on 20 topics. The experiment relies on the fact that the corpus is split into these 20 true classes. We cluster the corpus with the intent to capture these classes as clusters. We control the number of clusters as a model hyperparameter, fitting models with with 5, 10, 20, 40, and 80 clusters. We hypothesize that vector search precision and recall will be higher the closer the number of clusters is to the true 20 classes. 

## Replicating the experiment
TODO
