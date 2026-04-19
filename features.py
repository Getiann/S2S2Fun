import numpy as np
import torch
import glob
import os
from functools import lru_cache
from Bio.SeqUtils import seq1
from Bio.PDB import PDBParser
from Bio import PDB
import pandas as pd


def cif2pdb(cif_file_path, pdb_file_path):
    if not os.path.exists(cif_file_path):
        raise FileNotFoundError(f"CIF file not found: {cif_file_path}")
    if os.path.exists(pdb_file_path):
        os.remove(pdb_file_path)
    parser = PDB.MMCIFParser()
    structure = parser.get_structure('protein', cif_file_path)
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(pdb_file_path)


def extract_sequences(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("PDB", pdb_file)
    sequences = {}
    for model in structure:
        for chain in model:
            residues = [res for res in chain if res.id[0] == ' ']
            sequences[chain.id] = "".join(seq1(res.resname) for res in residues)
    return sequences


def get_nearest_residues(pdb_file, ligand_resname="RET", n_residues=60):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    ligand_atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname().strip() == ligand_resname:
                    for atom in residue:
                        ligand_atoms.append(atom.coord)
    ligand_coords = np.array(ligand_atoms)

    if len(ligand_coords) == 0:
        raise ValueError(f"Ligand {ligand_resname} not found in {pdb_file}")

    residue_distances = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                atom_coords = np.array([atom.coord for atom in residue])
                dists = np.linalg.norm(atom_coords[:, None, :] - ligand_coords[None, :, :], axis=2)
                residue_distances.append(((chain.id, residue.id[1]), np.min(dists)))

    residue_distances.sort(key=lambda x: x[1])
    return [res[0][1] for res in residue_distances[:n_residues]]


@lru_cache(maxsize=None)
def cached_extract_sequences(pdb_file):
    return extract_sequences(pdb_file)


@lru_cache(maxsize=None)
def cached_get_nearest_residues(pdb_file, ligand_resname, n_residues):
    return get_nearest_residues(pdb_file, ligand_resname=ligand_resname, n_residues=n_residues)


def extract_features(pair, single, N_protein, N_ligand):
    pair = torch.as_tensor(pair, dtype=torch.float32)
    single = torch.as_tensor(single, dtype=torch.float32)

    pp_mean = pair[:N_protein, :N_protein, :].mean(dim=(0, 1))
    ll_mean = pair[N_protein:, N_protein:, :].mean(dim=(0, 1))
    pl_mean = pair[:N_protein, N_protein:, :].mean(dim=(0, 1))
    lp_mean = pair[N_protein:, :N_protein, :].mean(dim=(0, 1))
    pair_feat = torch.cat([pp_mean, ll_mean, pl_mean, lp_mean], dim=0)

    single_feat = torch.cat([
        single[:N_protein].mean(dim=0),
        single[N_protein:].mean(dim=0)
    ], dim=0)

    return torch.cat([pair_feat, single_feat], dim=0)


def get_embeddings_file(af3_out_dir, all_seq_ids=None):
    af3_embeddings_files = []
    for rank_file in glob.glob(f'{af3_out_dir}/*/*_ranking_scores.csv'):
        seq_id = rank_file.split('/')[-2]
        if all_seq_ids is not None and seq_id not in all_seq_ids:
            continue
        df = pd.read_csv(rank_file)
        best_seed = df.loc[df['ranking_score'].idxmax(), 'seed']
        embedding_file = glob.glob(os.path.dirname(rank_file) + f'/seed-{best_seed}_embeddings/*.npz')[0]
        af3_embeddings_files.append(embedding_file)
    return af3_embeddings_files
