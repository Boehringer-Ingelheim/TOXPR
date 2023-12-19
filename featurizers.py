import torch
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import DataStructs
from typing import Any
from molfeat.calc.bond import EdgeMatCalculator
from molfeat.calc.atom import AtomCalculator
from molfeat.trans.graph import AdjGraphTransformer


class GraphFeaturizer:
    """
    Generated node features:
    - atom_one_hot
    - atom_degree_one_hot
    - atom_implicit_valence_one_hot
    - atom_hybridization_one_hot
    - atom_is_aromatic
    - atom_formal_charge
    - atom_num_radical_electrons
    - atom_is_in_ring
    - atom_total_num_H_one_hot
    - atom_chiral_tag_one_hot
    - atom_is_chiral_center'

    Generated edge features:
    - bond_type_one_hot
    - bond_stereo_one_hot
    - bond_is_in_ring
    - bond_is_conjugated
    - bond_direction_one_hot

    """

    def __init__(self) -> None:
        self.adj_trans = AdjGraphTransformer(
            atom_featurizer=AtomCalculator(),
            bond_featurizer=EdgeMatCalculator(),
            explicit_hydrogens=False,
            self_loop=True,
            canonical_atom_order=True,
            dtype=torch.float,
        )

    def __call__(self, smiles) -> Any:
        try:
            if smiles is None:
                return None
            features = self.adj_trans(smiles)
            graph, atom_x, bond_x = features[0]
            graph, atom_x, bond_x = graph.numpy(), atom_x.numpy(), bond_x.numpy()

            # Limitation of pyspark - cannot store multidimensional arrays with different shapes
            # Therefore we store flatten the arrays and store the shapes for restoration
            shape_g, shape_a, shape_b = list(graph.shape), list(
                atom_x.shape), list(bond_x.shape)

            return [
                graph.flatten().tolist(),
                atom_x.flatten().tolist(),
                bond_x.flatten().tolist(),
                list(map(float, shape_g)),
                list(map(float, shape_a)),
                list(map(float, shape_b)),
            ]
        except ValueError:
            return None


class MorganFeaturizer:
    def __init__(self, radius=2, n_bits=1024) -> None:
        self.radius = radius
        self.n_bits = n_bits

    def __call__(self, smiles) -> Any:
        if smiles is None:
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp_vec = Chem.AllChem.GetMorganFingerprintAsBitVect(
            mol, self.radius, self.n_bits)
        arr = np.unpackbits(np.frombuffer(DataStructs.BitVectToBinaryText(
            fp_vec), dtype=np.uint8), bitorder="little")
        return arr.tolist()
