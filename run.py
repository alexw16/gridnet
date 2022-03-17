import os
import numpy as np
import scanpy as sc
import argparse
import time

from models import *
from gridnet import *
from utils import *

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-m','--method',dest='method',type=str,default='baseline')
	parser.add_argument('-d','--dataset',dest='dataset',type=str,default=1)
	parser.add_argument('-nn','--n_neighbors',dest='n_neighbors',type=int,default=15)
	parser.add_argument('-nl','--n_layers',dest='n_layers',type=int,default=5)
	parser.add_argument('-dev','--device',dest='device',type=int,default=0)

	args = parser.parse_args()

	print('Loading data...')
	print('Dataset:',args.dataset)
	print(args.method)

	base_dir = '/data/cb/alexwu/mm_finemap'
	data_dir = '{}/datasets/{}'.format(base_dir,args.dataset)
	save_dir = '{}/results/tests_nn/{}'.format(base_dir,args.dataset)

	sketch = 'sketch' in args.method
	rna_adata,atac_adata = load_multiome_data(data_dir,args.dataset,
		sketch=sketch,preprocess=True)
	atac_adata.var.index = atac_adata.var.index.astype(int)

	torch.cuda.empty_cache()

	if 'trial' in args.method:
		seed = int(args.method.split('trial')[1][0])
	else:
		seed = 1
	print('Seed',seed)

	# load indices
	atac_idx = np.loadtxt(os.path.join(data_dir,'atac_idx.txt'),dtype=int)
	rna_idx = np.loadtxt(os.path.join(data_dir,'rna_idx.txt'),dtype=int)

	print('{} examples'.format(len(rna_idx)))

	if 'baseline' in args.method:

		print('Pseudotime inference')
		sc.pp.neighbors(rna_adata,use_rep='X_schema',n_neighbors=args.n_neighbors)
		sc.tl.dpt(rna_adata)

		sorted_inds = np.argsort(rna_adata.obs['dpt_pseudotime'].values)

		rna_X = rna_adata.X.toarray()[sorted_inds]
		atac_X = atac_adata.X.toarray()[sorted_inds]

		if 'bin' in args.method:
			print('binning...')
			bin_size = int(rna_X.shape[0]/100)
			rna_X = np.array([rna_X[i:i+bin_size].mean(0) 
				for i in range(0,rna_X.shape[0],bin_size)])
			atac_X = np.array([atac_X[i:i+bin_size].mean(0) 
				for i in range(0,atac_X.shape[0],bin_size)])

		start = time.time() 
		granger = []
		for gene_ind,atac_ind in zip(*[rna_idx,atac_idx]):
			data = np.array([rna_X[:,gene_ind],atac_X[:,atac_ind]]).T
			granger.append(run_granger(data))

			if len(granger) % 1000 == 0:
				print(len(granger)/len(atac_idx),time.time()-start,'seconds',np.array(granger).mean())

		granger = np.array(granger)

		print('Total Time: {} seconds'.format(time.time()-start))

		np.savetxt(os.path.join(save_dir,'{}.bin.results.txt'.format(args.method)),granger)

	elif 'GVAR' in args.method or 'neuralGC' in args.method:

		print('Pseudotime inference')
		sc.pp.neighbors(rna_adata,use_rep='X_schema',n_neighbors=args.n_neighbors)
		sc.tl.dpt(rna_adata)

		# sort by pseudotime
		sorted_inds = np.argsort(rna_adata.obs['dpt_pseudotime'].values)
		rna_X = rna_adata.X.toarray()[sorted_inds]
		atac_X = atac_adata.X.toarray()[sorted_inds]

		# average data within equally sized bins
		if 'bin' in args.method:
			print('binning...')
			bin_size = int(rna_X.shape[0]/100)
			rna_X = np.array([rna_X[i:i+bin_size].mean(0) 
				for i in range(0,rna_X.shape[0],bin_size)])
			atac_X = np.array([atac_X[i:i+bin_size].mean(0) 
				for i in range(0,atac_X.shape[0],bin_size)])

		gene_peaks_dict = {}
		for gene_ind,atac_ind in zip(*[rna_idx,atac_idx]):
			if gene_ind not in gene_peaks_dict:
				gene_peaks_dict[gene_ind] = []
			gene_peaks_dict[gene_ind].append(atac_ind)

		device = "cuda" if torch.cuda.is_available() else "cpu"
		torch.cuda.set_device(args.device)

		start = time.time() 
		peak_gene_score_dict = {}
		for i,gene_ind in enumerate(sorted(list(gene_peaks_dict.keys()))):
			atac_inds_list = gene_peaks_dict[gene_ind]
			data = np.concatenate([rna_X[:,[gene_ind]],atac_X[:,atac_inds_list]],axis=1)

			if 'GVAR' in args.method:
				scores_list = run_GVAR(data,seed=seed,batch_size=data.shape[0])
			elif 'neuralGC' in args.method:
				scores_list = run_neuralGC(data,device=device,seed=seed)
				print(scores_list.mean(),len(scores_list))

			for atac_ind,score in zip(*[atac_inds_list,scores_list]):
				peak_gene_score_dict[(atac_ind,gene_ind)] = score

			if i % 100 == 0:
				print(i/len(gene_peaks_dict),time.time()-start,'seconds',scores_list.mean())

		granger = np.array([peak_gene_score_dict[(atac_ind,gene_ind)] 
				for gene_ind,atac_ind in zip(*[rna_idx,atac_idx])])

		print('Total Time: {} seconds'.format(time.time()-start))

		np.savetxt(os.path.join(save_dir,'{}.bin.results.txt'.format(args.method)),granger)

	elif 'graph' in args.method:

		per_pair = 'per_pair' in args.method
		train_separate = 'train_separate' in args.method

		rna_features = rna_adata.var.index.values[sorted(list(set(rna_idx)))]
		atac_features = atac_adata.var.index.values[sorted(list(set(atac_idx)))]

		rna_X,atac_X,rna_idx,atac_idx = retain_desired_indices(rna_adata,atac_adata,rna_idx,atac_idx)
		candidate_XY_pairs = [(atac_features[j],rna_features[i]) for i,j in zip(*[rna_idx,atac_idx])]

		iroot = rna_adata.uns['iroot']
		joint_feature_embeddings = rna_adata.obsm['X_schema']
		save_name = '{}.{}layers.nn{}'.format(args.method,args.n_layers,
			args.n_neighbors)

		# construct DAG
		dag_adjacency_matrix = construct_dag(joint_feature_embeddings,iroot,
			n_neighbors=args.n_neighbors,pseudotime_algo='dpt')

		run_gridnet(atac_X,rna_X,atac_features,rna_features,candidate_XY_pairs,
			dag_adjacency_matrix=dag_adjacency_matrix,n_layers=args.n_layers,
			device=args.device,save_dir=save_dir,save_name=save_name,shuffle=True,
			per_pair=per_pair,train_separate=train_separate)

