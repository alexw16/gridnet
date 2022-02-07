# GrID-Net
**GrID-Net** (**Gr**anger **I**nference on **D**AGs) is a graph neural network framework for Granger causal inference on directed acyclic graph (DAG)-structured dynamical systems. It takes as input 1) a dataset containing candidate Granger causal variables, 2) a dataset containing target variables, 3) a set of candidate Granger causal relationships to be evaluated, and 4) (optional) a DAG representing the relationship between observations in the datasets. The output is a table of statistics for each tested candidate Granger causal relationship that describes the significance of each of these relationships. 

## Usage
Here is a general workflow for applying GrID-Net to a dataset with a user-defined DAG. First, define the datasets containing the candidate Granger causal variables and the target variables. The rows of the two datasets should be paired, such that row *n* corresponds to the candidate Granger causal variable values and target variable values for the same observation *n*.  
In the case of inferring peak-gene pairs from single-cell multi-omic datasets explored in our paper, the candidate Granger causal variables correspond to peaks and the target variables correspond to genes, with each cell as a unique observation. 

```python
X = numpy.ndarray with dimensions N x C (N = number of observations, C = number of candidate Granger causal variables)
Y = numpy.ndarray with dimensions N x T (N = number of observations, T = number of target variables)
X_feature_names = [list of string] of length C
Y_feature_names = [list of string] of length T
```

Then, define the set of candidate Granger causal relationships to be evaluated, using the names of the Granger causal variables and target variables defined above.
```python
candidate_XY_pairs = [list of tuples (x,y)] (x = name of Granger causal variable, y = name of target variable)
```

GrID-Net requires a user-defined DAG that defines the relationships between observations in the dynamical system of interest. The rows and columns of the adjacency matrix that represents the DAG should be ordered in accordance with the order of observations used in defining ```X``` and ```Y``` above. 
```python
custom_DAG = numpy.ndarray with dimensions N x N (N = number of observations)
```

Once these various inputs are defined, simply run the line below to train the GrID-Net model and evaluate the candidate Granger causal relationships. 
```python
import gridnet

gridnet(X,Y,X_feature_names,Y_feature_names,candidate_XY_pairs,custom_DAG=custom_DAG)
```

## Questions
For questions about the code, contact [alexwu@mit.edu](mailto:alexwu@mit.edu).
