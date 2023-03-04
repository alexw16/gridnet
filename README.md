![GrID-Net Schematic](https://user-images.githubusercontent.com/6614489/222857807-263d0fa0-c7e9-41e2-aa61-2b1ba5f59df2.png)

# GrID-Net
**GrID-Net** (**Gr**anger **I**nference on **D**AGs) is a graph neural network framework for Granger causal inference on directed acyclic graph (DAG)-structured dynamical systems, as described in the preprint ["An econometric lens resolves cell-state parallax"](https://www.biorxiv.org/content/10.1101/2023.03.02.530529) and the paper ["Granger causal inference on DAGs identifies genomic loci regulating transcription"](https://arxiv.org/abs/2210.10168). It takes as input (1) a dataset featurized by the set of candidate Granger causal variables of interest, (2) a corresponding dataset featurized by the set of target variables of interest, (3) a set of candidate Granger causal relationships to be evaluated, and (4) a DAG representing the relationship between observations in the dataset. The output is a table of statistics that describes the significance of each of the tested candidate Granger causal relationships. 

## Installation
```python
pip install gridnet_learn  
```

Then, in Python
```python
import gridnet
```

## Usage
### Quick Start: Multimodal Single-Cell Data (RNA-seq + ATAC-seq)
To use GrID-Net for inferring noncoding locus (i.e. peak)-gene links from single-cell multimodal data, we have provided a wrapper function that minimally requires (1) unprocessed AnnData objects for each of the RNA-seq and ATAC-seq datasets and (2) either a marker gene for the root cell type or a root cell index for pseudotime inference. This function will automatically perform pre-processing of the RNA-seq and ATAC-seq datasets, determine the candidate peak-gene pairs to evaluate, construct the DAG of cells, and run GrID-Net. 

```python
# runs GrID-Net using a marker gene to determine the root cell 
results_df = gridnet.gridnet_multimodal(rna_adata,atac_adata,root_cell_marker_gene='TOP2A')

# runs GrID-Net given the index of the root cell 
results_df = gridnet.gridnet_multimodal(rna_adata,atac_adata,root_cell_idx=0)
```

Note that the AnnData object for the RNA-seq dataset must include annotations for the genomic position of the transcription start site (TSS) for each gene in its ```adata.var``` DataFrame. Similarly, the AnnData object for the ATAC-seq dataset must include annotations for the genomic positions of the start and end sites for all peaks in its ```adata.var``` DataFrame. The default settings require that the columns in the DataFrames corresponding to these annotations be labeled as such.
 
| RNA-seq | ATAC-seq |
| :--- | :--- |
| gene TSS chromosome: ```chr_no``` | peak chromsome: ```chr_no``` |
| gene TSS position: ```txstart```  | peak start: ```start``` |
|                                   | peak end: ```end``` |

Some key parameters that users can optionally adjust in this function include 
1. ```distance_thresh```: maximum genomic distance between a peak and a gene for consideration as a candidate peak-gene pair (default: 1 Mb)
2. ```rna_filter_gene_percent```: minimum proportion of cells a gene is expressed to be considered in a candidate peak-gene pair (default: 1% of cells)
3. ```atac_filter_peak_percent```: minimum proportion of cells a peak is accessible to be considered in a candidate peak-gene pair (default: 0.1% of cells)

### Step-by-Step: Multimodal Single-Cell Data (RNA-seq + ATAC-seq)
For users seeking greater control over the various pre-processing and analysis steps for single-cell multimodal data, we include here the key steps that lead up to using GrID-Net on this data.
1. Data pre-processing: count normalization and log(1+x) transformation
```python
gridnet.preprocess_multimodal(rna_adata,atac_adata)
```
2. Determining candidate peak-gene pairs: identify all peak-gene pairs to be evaluated based on the genomic distance between a peak and the TSS of a gene. Outputs a DataFrame containing the candidate peak-gene pairs.
```python
candidates_df = gridnet.identify_all_peak_gene_link_candidates(rna_adata,atac_adata,distance_thresh=1e6)
```
3. Learn multimodal cell representations: use Schema to learn a joint representation for each cell that unifies information from both the ATAC-seq and RNA-seq modalities
```python
X_joint = gridnet.schema_representations(rna_adata,atac_adata) 
```
4. Construct the DAG of cells: use the multimodal cell representations to infer pseudotime and to orient the edges of the kNN graph to form a DAG 
```python
dag_adjacency_matrix = gridnet.construct_dag(X_joint,iroot,n_neighbors=15,pseudotime_algo='dpt')
```
5. Run GrID-Net: uses the various results from above to infer Granger causal peak-gene links. Outputs a DataFrame annotating the candidate peak-gene pairs and the corresponding Granger causality test statistics.  
```python
X = atac_adata.X.toarray() if issparse(atac_adata.X) else atac_adata.X
Y = rna_adata.X.toarray() if issparse(rna_adata.X) else rna_adata.X
X_feature_names = atac_adata.var.index.values
Y_feature_names = rna_adata.var.index.values
candidate_XY_pairs = [(x,y) for x,y in candidates_df[['atac_id','gene']].values]
    
results_df = gridnet.run_gridnet(X,Y,X_feature_names,Y_feature_names,
                                 candidate_XY_pairs,dag_adjacency_matrix)
``` 

### General Usage (beyond single-cell genomics, for arbitrary DAG-structured Granger causal inference) 
Here is a general workflow for applying GrID-Net to a dataset with a user-defined DAG. First, prepare one dataset that contains the candidate Granger causal variables and another that contains the target variables. The rows of the two datasets should be paired, such that row *n* corresponds to the candidate Granger causal variable values and target variable values for the same observation *n*.  

In the case of inferring peak-gene pairs from single-cell multi-omic datasets, the candidate Granger causal variables correspond to peaks and the target variables correspond to genes, with each cell representing a unique observation. 

```python
# matrix of dimensions N x C 
# (N = number of observations, C = number of candidate Granger causal variables)
X : numpy.ndarray 

# matrix of dimensions N x T 
# (N = number of observations, T = number of target variables) 
Y : numpy.ndarray 

# list of length C
# (C = number of candidate Granger causal variables)
X_feature_names : [list of string] 

# list of length T
# (T = number of target variables)
Y_feature_names : [list of string] 
```

Then, define the set of candidate Granger causal relationships to be evaluated, using the names of the Granger causal variables and target variables defined above.

```python
# list of tuples (x,y)
# (x = name of Granger causal variable, y = name of target variable)
candidate_XY_pairs : [list of tuples] 
```

GrID-Net requires a user-defined DAG that defines the relationships between observations in the dynamical system of interest. The rows and columns of the adjacency matrix that represents the DAG should be ordered in accordance with the order of observations used in defining ```X``` and ```Y``` above. 

```python
# matrix of dimensions N x N
# N = number of observations
dag_adjacency_matrix : scipy.sparse.csr_matrix or numpy.ndarray 
```

For single-cell multimodal data, we construct a DAG of cell states using the below function. ```joint_feature_embeddings``` represent the cell embeddings used to assess cell-cell similarity. We use [Schema](https://github.com/rs239/schema) (Singh, R., Hie, B. *et al.*, 2021) to generate these joint feature embeddings. ```iroot``` corresponds to the index of the cell to be used as the root cell for pseudotime inference, ```n_neighbors``` specifies the number of nearest neighobrs to be used in the k-nearest neighbor graph, and ```pseudotime_algo``` denotes the pseudotime inference algorithm to be used for orienting the edges in the graph.  

```python
dag_adjacency_matrix = gridnet.construct_dag(joint_feature_embeddings,iroot,
                                             n_neighbors=15,pseudotime_algo='dpt')
```

Once these inputs are defined, simply run the line below to train the GrID-Net model and evaluate the candidate Granger causal relationships. 
```python

results_df = gridnet.run_gridnet(X,Y,X_feature_names,Y_feature_names,
                                 candidate_XY_pairs,dag_adjacency_matrix)
```

## Questions
For questions about the code, contact [alexwu@mit.edu](mailto:alexwu@mit.edu).
