# Pol.is

This repository includes examples for analyzing Polis conversation data in Jupyter notebooks. 

**Pol.is** is a real-time system for gathering, analyzing and understanding what large groups of people think in their own words, enabled by advanced statistics and machine learning. Pol.is has been used all over the world by governments, academics, independent media and citizens, and is completely open source.

<br/>

## Table of Contents
---

- [Data](#data)
- [Contents](#contents)
- [Get started](#get-started)
- [Requirements](#requirements)
- [Links](#links)
- [Contact](#mailbox-contact)


<br/>

## Data
---

The data are stored in directory ``Data`` and  is available here for the following conversations:

- **uberx.taiwan**: Engagement with the government of Taiwan as part of the [vTaiwan](https://www.wired.co.uk/article/taiwan-sunflower-revolution-audrey-tang-g0v) participatory process which led to the successful regulation of Uber in Taiwan.
- **bowling-green.american-assembly**: Conversation run by the [American Assembly](https://americanassembly.org/) in [Bowling Green Kentucky](https://en.wikipedia.org/wiki/Bowling_Green,_Kentucky) regarding what's important to residents, and question the narrative of a "divided America" at the local/regional scope
- **operation-marching-orders.march-on**: Conversation run for [March On](https://www.wearemarchon.org/)
  (sister organization to [Women's March](https://womensmarch.com/)) seeking organization direction and platform.
- **15-per-hour-seattle**: Polis demo  regarding opinions on the move
  to a $15/hr minimum wage.
- **football-concussions**: Polis demo conversation regarding the societal implications of brain injuries in American football.
- **brexit-consensus**: Labour Party test discussion regarding Brexit
- **canadian-electoral-reform**: Conversation regarding Canadian electoral reform.
- **biodiversity**: Scoop & PEP run [hivemind conversation regarding Biodiversity in NZ]


More datasets can be found in: https://github.com/compdemocracy/openData

<br/>

## Contents
---

- ``01. Community detection (kNN approach).ipynb``: Community detection using  a $k$-neighbor graph of the participant-votes matrix is given as input to the Leiden or Louvain algorithm. 

- ``02. Community detction.ipynb``: Update of the previous implementation, which
performs community detection using a $k$-neighbor graph of the participant-votes matrix as input to the Leiden or Louvain algorithm. However, the participant-votes matrix can be given directly. Quality of results is low, if the adjacency matrix is given as-is, in one-layer format. That is, as a weighted adjacency matrix with 1.0 for positive links, and -1.0 for negative ones. Quality of results is comparable to k-neighbors graph, if the two-layer structure is explicitly given as input to the Leiden algorithm. That is, the weighted adjacency matrix is split in two, G_pos and G_neg. G_pos is an unweighted version of the same matrix, with 1.0 where positive links exists and 0.0 otherwise. G_neg is defined similarly for the negative links. Both layers are given as input to Leiden, with layer_weights=[1 -1]  (i.e. preferring positive connections and penalizing negative ones.)


- ``03. Opinion group.ipynb``: To form opinion groups, we take the PCA coordinates and perform K-means clustering with K=100. 
These fine-graine cluster serve as the basis for a more coarse-grained clustering, also using $k$-means. In fact, we take the 100 centers (obtained from the first K-means clustering), and run additional K-means clustering for K ranging between 2 and 5. The $k$ for which the silhoutte coefficient (a measure of withing-cluster similarity vs. between-cluster dissimilarity) is optimal is chosen for the opinion groups.

<br/>

## Get started
--- 

1. Create a virtual environment 
```
    conda create -n polis python=3.8
```

2. Activate the virtual environment 
```
    conda activate polis
```
3. Install requirements 
```
    pip install -r requirements.txt
```
4. Run jupyter notebooks

<br/>

## Requirements
---

- python>=3.8
- seaborn==0.11.1
- scikit-learn==0.24.2
- pandas==1.1.3
- numpy==1.20.3
- matplotlib==3.4.2
- louvain==0.7.1
- leidenalg==0.8.9
- igraph==0.9.9
- altair==4.1.0

<br/>


## Links
---

- [Pol.is website](https://pol.is/home)
- [Basic information](https://compdemocracy.org/Welcome/)
- [Algorithms](https://compdemocracy.org/algorithms/)


<br/>

## :mailbox: Contact
---

Ioannis E. Livieris (livieris@novelcore.eu)