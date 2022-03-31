import numpy as np
import os
import torch
from scipy.stats import f
from scipy.sparse import csr_matrix
import scanpy as sc
import scanpy.external as sce
from anndata import AnnData

def construct_dag(joint_feature_embeddings,iroot,n_neighbors=15,pseudotime_algo='dpt'):
	
	"""Constructs the adjacency matrix for a DAG.
	Parameters
	----------
	joint_feature_embeddings: 'numpy.ndarray' (default: None)
		Matrix of low dimensional embeddings with rows corresponding
		to observations and columns corresponding to feature embeddings
		for constructing a DAG if a custom DAG is not provided.
	iroot: 'int' (default: None)
		Index of root cell for inferring pseudotime for constructing a DAG 
		if a custom DAG is not provided.
	n_neighbors: 'int' (default: 15)
		Number of nearest neighbors to use in constructing a k-nearest
		neighbor graph for constructing a DAG if a custom DAG is not provided.
	pseudotime_algo: {'dpt','palantir'} 
		Pseudotime algorithm to use for constructing a DAG if a custom DAG 
		is not provided. 'dpt' and 'palantir' perform the diffusion pseudotime
		(Haghverdi et al., 2016) and Palantir (Setty et al., 2019) algorithms, 
		respectively.
	"""

	pseudotime,knn_graph = infer_knngraph_pseudotime(joint_feature_embeddings,iroot,
		n_neighbors=n_neighbors,pseudotime_algo=pseudotime_algo)
	dag_adjacency_matrix = dag_orient_edges(knn_graph,pseudotime)

	return dag_adjacency_matrix

def infer_knngraph_pseudotime(joint_feature_embeddings,iroot,n_neighbors=15,pseudotime_algo='dpt'):

	adata = AnnData(joint_feature_embeddings)
	adata.obsm['X_joint'] = joint_feature_embeddings
	adata.uns['iroot'] = iroot

	if pseudotime_algo == 'dpt':
		sc.pp.neighbors(adata,use_rep='X_joint',n_neighbors=n_neighbors)
		sc.tl.dpt(adata)
		adata.obs['pseudotime'] = adata.obs['dpt_pseudotime'].values
		knn_graph = adata.obsp['distances'].astype(bool).astype(float)
	elif pseudotime_algo == 'palantir':
		sc.pp.neighbors(adata,use_rep='X_joint',n_neighbors=n_neighbors)
		sce.tl.palantir(adata, knn=n_neighbors,use_adjacency_matrix=True,
			distances_key='distances')
		pr_res = sce.tl.palantir_results(adata,
			early_cell=adata.obs.index.values[adata.uns['iroot']],
			ms_data='X_palantir_multiscale')
		adata.obs['pseudotime'] = pr_res.pseudotime
		knn_graph = adata.obsp['distances'].astype(bool).astype(float)

	return adata.obs['pseudotime'].values,knn_graph

def dag_orient_edges(adjacency_matrix,pseudotime):

	A = adjacency_matrix.astype(bool).astype(float)
	D = -1*np.sign(pseudotime[:,None] - pseudotime).T
	D = (D == 1).astype(float)
	D = (A.toarray()*D).astype(bool).astype(float)

	return D

def construct_S0_S1(D):

	D_0 = D.copy()
	D_1 = D.copy() + np.eye(D.shape[0])
	S_0 = D_0.copy()
	S_1 = D_1.copy()

	D_0_sum = D_0.sum(1)
	D_0_sum[D_0_sum == 0] = 1
	S_0 = (S_0.T/D_0_sum)
	S_1 = (S_1.T/D_1.sum(1))

	return S_0,S_1

def load_multiome_data(data_dir,dataset,sampling=None,preprocess=True):

	if sampling == 'geosketch':
		atac_adata = sc.read(os.path.join(data_dir,
			'{}.atac.sketch.h5ad'.format(dataset)))
		rna_adata = sc.read(os.path.join(data_dir,
			'{}.rna.sketch.h5ad'.format(dataset)))
	elif sampling == 'uniform':
		atac_adata = sc.read(os.path.join(data_dir,
			'{}.atac.uniform.h5ad'.format(dataset)))
		rna_adata = sc.read(os.path.join(data_dir,
			'{}.rna.uniform.h5ad'.format(dataset)))
	else:
		atac_adata = sc.read(os.path.join(data_dir,
			'{}.atac.h5ad'.format(dataset)))
		rna_adata = sc.read(os.path.join(data_dir,
			'{}.rna.h5ad'.format(dataset)))

	if preprocess:

		# scale by maximum 
		# (rna already normalized by library size + log-transformed)
		X_max = rna_adata.X.max(0).toarray().squeeze()
		X_max[X_max == 0] = 1
		rna_adata.X = csr_matrix(rna_adata.X / X_max)

		# atac: normalize library size + log transformation
		sc.pp.normalize_total(atac_adata,target_sum=1e4)
		sc.pp.log1p(atac_adata)

	return rna_adata,atac_adata

def retain_desired_indices(rna_adata,atac_adata,rna_idx,atac_idx):

	atac_X = atac_adata.X[:,sorted(list(set(atac_idx)))].toarray()
	rna_X = rna_adata.X[:,sorted(list(set(rna_idx)))].toarray()

	sorted_atac_idx = sorted(list(set(atac_idx)))
	sorted_rna_idx = sorted(list(set(rna_idx)))

	atac_idx_map = {idx1:idx2 for idx1,idx2 in \
		zip(*[sorted(list(set(atac_idx))),range(atac_X.shape[1])])}
	rna_idx_map = {idx1:idx2 for idx1,idx2 in \
		zip(*[sorted(list(set(rna_idx))),range(rna_X.shape[1])])}

	atac_idx = np.array([atac_idx_map[idx] for idx in atac_idx])
	rna_idx = np.array([rna_idx_map[idx] for idx in rna_idx])

	return rna_X,atac_X,rna_idx,atac_idx

def rss_ratio_ftest(rss_ratio,n_layers,n):
	
	from scipy.stats import f

	# number of reduced model parameters 
	# (W,b for each RNA layer)
	p1 = n_layers * 2 
	# number of full model parameters 
	# (W,b for each layer of RNA, ATAC + interaction)
	p2 = n_layers * 2 * 2 + 1 
	
	dfn = 2 * n_layers + 1
	dfd = n - 4 * n_layers - 1

	F = (rss_ratio - 1) * (n - p2)/(p2-p1)

	return 1-f.cdf(F,dfn,dfd)

def calculate_fdr(p):
	return fdrcorrection(p)[1]
