# GrID-Net
**GrID-Net** (**Gr**anger **I**nference on **D**AGs) is a graph neural network framework for Granger causal inference on directed acyclic graph (DAG)-structured dynamical systems, as described in the paper ["Granger causal inference on DAGs identifies genomic loci regulating transcription"](https://openreview.net/forum?id=nZOUYEN6Wvy). It takes as input (1) a dataset featurized by the set of candidate Granger causal variables of interest, (2) a corresponding dataset featurized by the set of target variables of interest, (3) a set of candidate Granger causal relationships to be evaluated, and (4) a DAG representing the relationship between observations in the dataset. The output is a table of statistics that describes the significance of each of the tested candidate Granger causal relationships. 

## Usage
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
dag_adjacency_matrix = construct_dag(joint_feature_embeddings,iroot,n_neighbors=15,pseudotime_algo='dpt')
```

Once these inputs are defined, simply run the line below to train the GrID-Net model and evaluate the candidate Granger causal relationships. 
```python
import gridnet

results_df = run_gridnet(X,Y,X_feature_names,Y_feature_names,
                         candidate_XY_pairs,dag_adjacency_matrix)
```

## Questions
For questions about the code, contact [alexwu@mit.edu](mailto:alexwu@mit.edu).
