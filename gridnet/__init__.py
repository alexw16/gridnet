__version__ = "0.1.0"
__citation__ = """Singh, Wu, Berger. "Granger causal inference on DAGs identifies genomic loci regulating transcription."  ICLR 2022. 
 Wu, Singh, Walsh, Berger. "An econometric lens resolves cell-state parallax." bioRxiv."""
from . import (
    gridnet,
    utils,
)

from utils import construct_dag, infer_knngraph_pseudotime, dag_orient_edges, load_multiome_data
from gridnet import run_gridnet

__all__ = [
    "construct_dag",
    "infer_knngraph_pseudotime",
    "dag_orient_edges",
    "load_multiome_data",
    "run_gridnet",
]
