import torch
import torch.nn as nn
import numpy as np
import time
import pandas as pd
import os
import sys
from scipy.sparse import issparse

from .utils import *
from .models import GraphGrangerModule

def run_gridnet(X,Y,X_feature_names,Y_feature_names,candidate_XY_pairs,dag_adjacency_matrix,
			n_layers=10,device=0,seed=1,shuffle=True,batch_size=1024,optim='adam',
			initial_learning_rate=0.001,beta_1=0.9,beta_2=0.999,max_epochs=20,
			train_separate=False,save_dir='./gridnet_results',save_name='gridnet',
			verbose=True):

	"""Runs GrID-Net to infer Granger causal relationships from DAGs.
	Parameters
	----------
	X: `numpy.ndarray`
		Matrix of candidate Granger causal variables with rows corresponding
		to observations and columns corresponding to the variables.
	Y: `numpy.ndarray`
		Matrix of target variables with rows corresponding
		to observations and columns corresponding to the variables.
	X_feature_names: 'list' of 'string' 
		List of names of the candidate Granger causal variables
	Y_feature_names: 'list' of 'string' 
		List of names of the target variables.
	candidate_XY_pairs: 'list' of 'tuple'
		List of candidate Granger causal pairs (x,y) with x and y named
		according to X_feature_names and Y_feature_names.
	dag_adjacency_matrix: `scipy.sparse.csr_matrix` or `numpy.ndarray`
		Adjacency matrix of the DAG. 
	n_layers: 'int' (default: 10)
		Number of layers to use in the graph neural network architecture.
	device: 'int' (default: 0)
		Device index to select.
	seed: 'int' (default: 1)
		Seed for random state generation.
	shuffle: 'bool' (default: True)
		Shuffle list of candidate_XY_pairs during training when True.
	batch_size: 'int' (default: 1024)
		Batch size.
	optim: {'adam','sgd'} (Default: 'adam')
		Optimization algorithm to use for training the model.
	initial_learning_rate: 'float' (default: 0.001)
		Initial learning rate to be used in optimization.
	beta_1: 'float' (default: 0.9)
		beta_1 term used in the Adam optimization algorithm.
	beta_2: 'float' (default: 0.999)
		beta_2 term used in the Adam optimization algorithm.
	max_epochs: 'int' (default: 20)
		Maximum number of epochs to train the model.
	train_separate: 'bool' (default: False)
		Train full and reduced models separately when True.
	save_dir: 'string' (default: './gridnet_results')
		Directory to save the model and the statistics associated
		with Granger causal analyses.
	save_name: 'string' (default: 'gridnet')
		Prefix to be used for naming the saved files.
	verbose: 'bool' (default: True)
		Prints logging output when True.
	"""

	if device != 'cpu':
		torch.cuda.set_device(device)
	torch.manual_seed(seed)
	np.random.seed(1)

	start = time.time()

	X_feature_names_idx_dict = {n: i for i,n in enumerate(X_feature_names)}
	Y_feature_names_idx_dict = {n: i for i,n in enumerate(Y_feature_names)}

	X_idx = np.array([X_feature_names_idx_dict[x] for (x,y) in candidate_XY_pairs])
	Y_idx = np.array([Y_feature_names_idx_dict[y] for (x,y) in candidate_XY_pairs])
	pairs_idx = np.arange(len(candidate_XY_pairs))

	if shuffle:
		shuffled_idx_list = np.arange(len(candidate_XY_pairs))
		np.random.shuffle(shuffled_idx_list)

	S_0,S_1 = construct_S0_S1(dag_adjacency_matrix)

	model = GraphGrangerModule(n_layers,X_idx,Y_idx,final_activation='exp')

	model.to(device)
	S_0 = torch.FloatTensor(S_0).to(device)
	S_1 = torch.FloatTensor(S_1).to(device)

	if optim == 'sgd':
		optimizer = torch.optim.SGD(params=model.parameters(), 
			lr=initial_learning_rate)
	elif optim == 'adam':
		optimizer = torch.optim.Adam(params=model.parameters(), 
			lr=initial_learning_rate, betas=(beta_1, beta_2))

	if shuffle:
		train_model(model,Y,X,Y_idx[shuffled_idx_list],X_idx[shuffled_idx_list],
					pairs_idx[shuffled_idx_list],optimizer,device,max_epochs,batch_size,
					criterion=nn.MSELoss(),early_stop=True,tol=0.1/len(candidate_XY_pairs),
					verbose=verbose,S_0=S_0,S_1=S_1,train_separate=False)
	else:
		train_model(model,Y,X,Y_idx,X_idx,pairs_idx,optimizer,device,max_epochs,batch_size,
					criterion=nn.MSELoss(),early_stop=True,tol=0.1/len(candidate_XY_pairs),
					verbose=verbose,S_0=S_0,S_1=S_1,train_separate=False)

	_,results_dict = run_epoch('Inference',Y,X,Y_idx,X_idx,pairs_idx,
		S_0,S_1,model,optimizer,device,batch_size,criterion=nn.MSELoss(),
		verbose=verbose,train=False,statistics=['rss_ratio','ftest'])

	if not os.path.exists(save_dir):
		os.mkdir(save_dir)

	model.to('cpu')
	torch.save(model.state_dict(),os.path.join(save_dir,'{}.model_weights.pth'.format(
		save_name)))

	results_df = pd.DataFrame(np.array(candidate_XY_pairs).astype(str),
		columns=['X_names','Y_names'])
	for k,result in results_dict.items():
		results_df[k] = result
	results_df.to_csv(os.path.join(save_dir,'{}.statistics.txt'.format(
		save_name)),sep='\t',index=False)

	if verbose:
		print('Total Time: {} seconds'.format(time.time()-start))

	return results_df

def train_model(model,rna_X,atac_X,rna_idx,atac_idx,pair_idx,\
				optimizer,device,num_epochs,batch_size,\
				criterion=nn.MSELoss(),early_stop=True,tol=0.01,verbose=True,\
				S_0=None,S_1=None,train_separate=False):

	for epoch_no in range(num_epochs):
		run_epoch(epoch_no,rna_X,atac_X,rna_idx,atac_idx,pair_idx,
			S_0,S_1,model,optimizer,device,batch_size,criterion=criterion,
			verbose=verbose,train=True,train_separate=train_separate)

def run_epoch(epoch_no,rna_X,atac_X,rna_idx,atac_idx,pair_idx,S_0,S_1,model,
			  optimizer,device=0,batch_size=1024,criterion=nn.MSELoss(),
			  verbose=True,train=True,statistics=None,train_separate=False):

	if statistics:
		results_dict = {k: [] for k in statistics}

	total_loss = 0
	batch_split = np.arange(0, len(rna_idx), batch_size)

	stats_criterion = nn.MSELoss(reduction='none')

	start = time.time()
	for i in range(len(batch_split)):
		
		batch_rna_x = rna_X[:,rna_idx[i*batch_size:batch_size*(i+1)]].T
		batch_atac_x = atac_X[:,atac_idx[i*batch_size:batch_size*(i+1)]].T
		
		batch_rna_x = torch.from_numpy(batch_rna_x).float().to(device)
		batch_atac_x = torch.from_numpy(batch_atac_x).float().to(device)

		batch_atac_idx = torch.from_numpy(atac_idx[i*batch_size:batch_size*(i+1)]).long().to(device)
		batch_rna_idx = torch.from_numpy(rna_idx[i*batch_size:batch_size*(i+1)]).long().to(device)
		batch_pair_idx = torch.from_numpy(pair_idx[i*batch_size:batch_size*(i+1)]).long().to(device)

		full_preds,red_preds = model(batch_atac_x,batch_rna_x,\
			batch_atac_idx,batch_rna_idx,batch_pair_idx,S_0,S_1)

		targets = batch_rna_x

		if train_separate == 'full' and not statistics:

			full_loss = criterion(full_preds, targets)

			if train:
				optimizer.zero_grad()
				full_loss.backward()
				optimizer.step()

			total_loss += full_loss.data.cpu().numpy()

		elif train_separate == 'reduced' and not statistics:

			red_loss = criterion(red_preds, targets)

			if train:
				optimizer.zero_grad()
				red_loss.backward()
				optimizer.step()

			total_loss += red_loss.data.cpu().numpy()

		else:
			full_loss = criterion(full_preds, targets)
			red_loss = criterion(red_preds, targets)

			base_loss =  full_loss + red_loss
			total_loss += base_loss.data.cpu().numpy()

			if train:
				optimizer.zero_grad()
				base_loss.backward()
				optimizer.step()

		if statistics:

			full_rss = stats_criterion(full_preds,targets).data.cpu().numpy()
			red_rss = stats_criterion(red_preds,targets).data.cpu().numpy()

			for j in range(full_rss.shape[0]):

				if 'rss_ratio' in statistics:
					rss_ratio = red_rss[j].sum()/full_rss[j].sum()
					results_dict['rss_ratio'].append(rss_ratio)

				if 'ftest' in statistics:
					# num of parameters: W,b per layer per modality
					rss_ratio = red_rss[j].sum()/full_rss[j].sum()
					ftest_p = rss_ratio_ftest(rss_ratio,
						model.num_hidden_layers,rna_X.shape[0])
					results_dict['ftest'].append(ftest_p)

	end = time.time()

	if verbose:
		mode = 'train' if train else 'test'
		print('Epoch {} ({:.2f} seconds): {} loss {:.2f}'.format(epoch_no,\
			end-start,mode,total_loss))

	if statistics:
		results_dict = {k: np.array(v) for k,v in results_dict.items()}
		return total_loss,results_dict
	else:
		return total_loss

def gridnet_multimodal(rna_adata,atac_adata,root_cell_marker_gene=None,root_cell_idx=None,
					   preprocess=True,rna_chr_key='chr_no',rna_txstart_key='txstart',
					   atac_chr_key='chr_no',atac_start_key='start',atac_end_key='end',
					   rna_normalize_total=1e5,atac_normalize_total=1e4,
					   rna_filter_gene_percent=0.01,atac_filter_peak_percent=0.001,
					   distance_thresh=1e6,scale_rna_by_max=True,device=0):
	
	"""Runs GrID-Net on multimodal single-cell data to infer Granger causal 
	peak-gene relationships. The default setting takes unnormalized counts data
	for both the RNA-seq and ATAC-seq datasets and automatically pre-processes
	both datasets. This wrapper function requires that the RNA-seq and ATAC-seq
	datasets be annotated with information about the genomic positions of gene
	transcription start sites and peak start and end sites.

	Parameters
	----------
	rna_adata: `AnnData`
		AnnData object containing the RNA-seq data. 
	atac_adata: `AnnData`
		AnnData object containing the ATAC-seq data.
	root_cell_marker_gene: 'str'
		Name of the marker gene for the root cell type to be used for
		pseudotime inference. One of 'root_cell_marker_gene' or 
		'root_cell_idx' must be provided.
	root_cell_idx: 'int'
		The index of the root cell to be used for pseudotime inference.
		One of 'root_cell_marker_gene' or 'root_cell_idx' must be provided.
	"""

	# data pre-processing
	if preprocess:
		print('Pre-processing multimodal data...')
		preprocess_multimodal(rna_adata,atac_adata,
					  rna_normalize_total=rna_normalize_total,
					  atac_normalize_total=atac_normalize_total,
					  rna_filter_gene_percent=rna_filter_gene_percent,
					  atac_filter_peak_percent=atac_filter_peak_percent)
	
	# set root cell for pseudotime inference
	if root_cell_idx is not None:
		iroot = root_cell_idx
	elif root_cell_marker_gene is None:
		if 'iroot' in rna_adata.uns:
			iroot = rna_adata.uns['iroot']
		elif 'iroot' in atac_adata.uns:
			iroot = atac_adata.uns['iroot']
		elif root_cell_idx is None:
			sys.stderr.write('ERROR: Root cell not annotated. ' 
							 'Please provide a marker gene for ' 
							 'the root cell type or provide the '
							 'index for the root cell.')
	else:      
		set_dpt_root(rna_adata,root_cell_marker_gene)
		iroot = rna_adata.uns['iroot']
	
	# identify candidate peak-gene links to be evaluated
	if rna_chr_key not in rna_adata.var \
		or rna_txstart_key not in rna_adata.var:
		sys.stderr.write('ERROR: RNA AnnData object annotations '
						 'for gene transcription start sites '
						 'do not match the provided keys.')  
	elif atac_chr_key not in atac_adata.var \
		or atac_start_key not in atac_adata.var \
		or atac_end_key not in atac_adata.var:
		sys.stderr.write('ERROR: ATAC AnnData object annotations '
						 'for peak locations do not match the '
						 'the provided keys.')
	else:
		print('Identifying all peak-gene link candidates...')
		candidates_df = identify_all_peak_gene_link_candidates(rna_adata,atac_adata,
							distance_thresh=distance_thresh,rna_chr_key=rna_chr_key,
							rna_txstart_key=rna_txstart_key,atac_chr_key=atac_chr_key,
							atac_start_key=atac_start_key,atac_end_key=atac_end_key)
	
	# learn joint multimodal representations
	print('Using Schema to learn joint multimodal representations...')
	X_joint = schema_representations(rna_adata,atac_adata)
	
	# construct DAG of cells
	print('Constructing DAG of cells...')
	dag_adjacency_matrix = construct_dag(X_joint,iroot,n_neighbors=15,
										 pseudotime_algo='dpt')
	
	# run GrID-Net
	print('Running GrID-Net...')
	if scale_rna_by_max:
		X_max = rna_adata.X.max(0).toarray().squeeze()
		X_max[X_max == 0] = 1
		rna_adata.X = csr_matrix(rna_adata.X / X_max)
		
	X = atac_adata.X.toarray() if issparse(atac_adata.X) else atac_adata.X
	Y = rna_adata.X.toarray() if issparse(rna_adata.X) else rna_adata.X

	X_feature_names = atac_adata.var.index.values
	Y_feature_names = rna_adata.var.index.values
	candidate_XY_pairs = [(x,y) for x,y in candidates_df[['atac_id','gene']].values]
	
	results_df = run_gridnet(X,Y,X_feature_names,Y_feature_names,
						 	candidate_XY_pairs,dag_adjacency_matrix,
						 	device=device)
	
	return results_df
