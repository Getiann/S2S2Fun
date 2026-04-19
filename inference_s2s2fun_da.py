import torch
import numpy as np
import pandas as pd
import glob
import os
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from models import FeatureExtractor, RankNetModel, DomainClassifier
from features import cif2pdb, cached_extract_sequences, cached_get_nearest_residues, extract_features


def _process_single_file(args):
    embeddings_file, p_num, ligand_num = args
    label_name = embeddings_file.split('/')[-3]
    data = np.load(embeddings_file, mmap_mode='r')
    pair   = data['pair_embeddings']
    single = data['single_embeddings']

    pdb_dir = os.path.dirname(os.path.dirname(embeddings_file))
    pdb_files = glob.glob(f'{pdb_dir}/*_model.pdb')
    if pdb_files:
        pdb_file = pdb_files[0]
    else:
        cif_file = glob.glob(f'{pdb_dir}/*.cif')[0]
        pdb_file = cif_file.replace('.cif', '.pdb')
        cif2pdb(cif_file, pdb_file)
        if not os.path.exists(pdb_file):
            raise FileNotFoundError(f'PDB file not found for {label_name} in {pdb_dir}')

    seqs = cached_extract_sequences(pdb_file)['A']
    all_tokens = len(seqs) + ligand_num

    if pair.shape[0] == all_tokens + 1:
        ligand_index = list(range(all_tokens - ligand_num, all_tokens + 1))
        ligand_index.remove(all_tokens - 5)
    else:
        ligand_index = list(range(all_tokens - ligand_num, all_tokens))

    near_index = [i - 1 for i in sorted(cached_get_nearest_residues(pdb_file, "RET", p_num))]
    index_np = np.array(near_index + ligand_index)

    feature = extract_features(
        torch.from_numpy(pair[index_np][:, index_np, :]),
        torch.from_numpy(single[index_np, :]),
        N_protein=p_num, N_ligand=ligand_num
    )
    return label_name, feature


def get_feature(af3_embeddings_files, p_num=30, ligand_num=20):
    n_files = len(af3_embeddings_files)
    feature_tensor = torch.empty((n_files, 1280), dtype=torch.float32)
    all_labels = [None] * n_files

    args_list = [(f, p_num, ligand_num) for f in af3_embeddings_files]
    with ThreadPoolExecutor(max_workers=64) as executor:
        futures = {executor.submit(_process_single_file, args): idx for idx, args in enumerate(args_list)}
        for future in as_completed(futures):
            idx = futures[future]
            seq_name, feature = future.result()
            all_labels[idx] = seq_name
            feature_tensor[idx] = feature

    return all_labels, feature_tensor


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    Gf = FeatureExtractor(dim=1280, hidden=512, dropout=0.1).to(device)
    Gc = RankNetModel(input_dim=1280).to(device)
    Gd = DomainClassifier(input_dim=1280).to(device)

    Gf.load_state_dict(checkpoint['Gf_state_dict'])
    Gc.load_state_dict(checkpoint['Gc_state_dict'])
    Gd.load_state_dict(checkpoint['Gd_state_dict'])

    Gf.eval(); Gc.eval(); Gd.eval()
    return Gf, Gc, Gd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="./mrp_s2s2fun_da.pt")
    parser.add_argument("--out_file", type=str, default="./output_da.csv")
    parser.add_argument("--af3_out_dir", type=str, required=True)
    args = parser.parse_args()

    p_num = 30
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    print('checkpoint:', args.checkpoint)

    all_pdb_files = glob.glob(f'{args.af3_out_dir}/*/*_model.cif')
    embedding_files = []
    for pdb_file in all_pdb_files:
        pdb_dir = os.path.dirname(pdb_file)
        emb = glob.glob(f'{pdb_dir}/seed-1_embeddings/*.npz')
        if emb:
            embedding_files.append(emb[0])

    all_labels, all_features = get_feature(embedding_files, p_num=p_num, ligand_num=20)

    Gf, Gc, Gd = load_model(args.checkpoint, device)
    Gf = torch.nn.DataParallel(Gf).to(device)
    Gc = torch.nn.DataParallel(Gc).to(device)

    with torch.no_grad():
        features_tensor = all_features.float().to(device)
        embeddings = Gf(features_tensor)
        scores = Gc(embeddings).cpu().numpy()

    df = pd.DataFrame({'seq_id': all_labels, 'score': scores.squeeze()})
    df.to_csv(args.out_file, index=False)
    print('Saved predictions to:', args.out_file)


if __name__ == "__main__":
    main()
