import torch
import torch.nn as nn
	
class GraphConvLayer(nn.Module):
	def __init__(self,n_vars):
		super(GraphConvLayer, self).__init__()
		
		self.W = nn.Parameter(torch.ones((n_vars,1)),requires_grad=True)
		self.b = nn.Parameter(torch.ones(n_vars,1),requires_grad=True)
		nn.init.xavier_normal_(self.W.data)
		nn.init.xavier_normal_(self.b.data)
		
	def forward(self,x,S,idx):
		return (x @ S) * self.W[idx] + self.b[idx]

class GraphGrangerModule(nn.Module):
	def __init__(self,num_hidden_layers,atac_idx,rna_idx,final_activation='exp'):
		super(GraphGrangerModule, self).__init__()

		self.num_hidden_layers = num_hidden_layers
		self.n_peaks = len(set(atac_idx))
		self.n_genes = len(set(rna_idx))
		self.n_pairs = len(atac_idx)

		self.final_activation = final_activation

		for n in range(num_hidden_layers):

			# full model
			setattr(self,'atac_gcn_{}'.format(n+1),GraphConvLayer(self.n_peaks))
			setattr(self,'rna_gcn_{}'.format(n+1),GraphConvLayer(self.n_genes))

			# reduced model
			setattr(self,'rna_gcn_r_{}'.format(n+1),GraphConvLayer(self.n_genes))

		self.C_a = nn.Parameter(torch.ones(self.n_pairs,1),requires_grad=True)
		nn.init.xavier_normal_(self.C_a.data)

		self.relu = nn.Tanh()
		self.sigmoid = nn.Sigmoid()
		
	def forward(self,atac_x,rna_x,atac_idx,rna_idx,pair_idx,S_0,S_1):
		
		atac_out_list = []
		rna_out_list = []
		rna_r_out_list = []

		atac_out = self.atac_gcn_1(atac_x,S_0,atac_idx)
		rna_out = self.rna_gcn_1(rna_x,S_0,rna_idx)
		rna_r_out = self.rna_gcn_r_1(rna_x,S_0,rna_idx)

		atac_out_list.append(atac_out)
		rna_out_list.append(rna_out)
		rna_r_out_list.append(rna_r_out)

		for n in range(1,self.num_hidden_layers):

			atac_out = self.relu(atac_out)
			rna_out = self.relu(rna_out)
			rna_r_out = self.relu(rna_r_out)

			atac_out = getattr(self,'atac_gcn_{}'.format(n+1))(atac_out,S_1,atac_idx)
			rna_out = getattr(self,'rna_gcn_{}'.format(n+1))(rna_out,S_1,rna_idx)
			rna_r_out = getattr(self,'rna_gcn_r_{}'.format(n+1))(rna_r_out,S_1,rna_idx)

			atac_out_list.append(atac_out)
			rna_out_list.append(rna_out)
			rna_r_out_list.append(rna_r_out)

		atac_out = torch.stack(atac_out_list,axis=0).mean(0)
		rna_out = torch.stack(rna_out_list,axis=0).mean(0)
		rna_r_out = torch.stack(rna_r_out_list,axis=0).mean(0)

		full_output = rna_out.squeeze() \
			+ atac_out.squeeze()*self.C_a[pair_idx] \

		reduced_output = rna_r_out.squeeze()

		if self.final_activation == 'exp':
			full_output = torch.exp(full_output)
			reduced_output = torch.exp(reduced_output)
		elif self.final_activation == 'sigmoid':
			full_output = self.sigmoid(full_output)
			reduced_output = self.sigmoid(reduced_output)

		return full_output,reduced_output
