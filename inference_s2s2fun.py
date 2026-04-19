import torch
import torch.nn as nn
import numpy as np
import glob
import os
import pandas as pd
import argparse

from models import RankNetModel
from features import (
    cif2pdb, extract_sequences, get_nearest_residues, extract_features
)


def get_embeddings_file(af3_out_dir):
    af3_embeddings_files = []
    for rank_file in glob.glob(f'{af3_out_dir}/*/*_ranking_scores.csv'):
        df = pd.read_csv(rank_file)
        best_seed = df.loc[df['ranking_score'].idxmax(), 'seed']
        embedding_file = glob.glob(os.path.dirname(rank_file) + f'/seed-{best_seed}_embeddings/*.npz')[0]
        af3_embeddings_files.append(embedding_file)
    return af3_embeddings_files


def get_feature(af3_out_dir, af3_embeddings_files, p_num=30, ligand_num=20):
    all_labels = []
    all_features = []
    for embeddings_file in af3_embeddings_files:
        seq_name = embeddings_file.split('/')[-3]
        data = np.load(embeddings_file)
        pair = data['pair_embeddings']
        single = data['single_embeddings']

        pdb_file = os.path.join(af3_out_dir, f"{seq_name.lower()}/{seq_name.lower()}_model.pdb")
        if not os.path.exists(pdb_file):
            cif_file = pdb_file.replace('.pdb', '.cif')
            cif2pdb(cif_file, pdb_file)

        seqs = extract_sequences(pdb_file)['A']
        all_tokens = len(seqs) + ligand_num
        ligand_index = list(range(all_tokens - ligand_num, all_tokens))

        near_index = [i - 1 for i in sorted(get_nearest_residues(pdb_file, ligand_resname="RET", n_residues=p_num))]
        index = np.array(near_index + ligand_index)

        sub_pair = pair[index][:, index, :]
        sub_single = single[index, :]

        assert sub_pair.shape[0] == sub_single.shape[0]
        assert sub_pair.shape[0] == p_num + ligand_num

        feature = extract_features(torch.tensor(sub_pair), torch.tensor(sub_single), p_num, ligand_num)
        all_labels.append(seq_name)
        all_features.append(feature)

    features_tensor = torch.stack(all_features)
    assert features_tensor.shape[0] == len(all_labels)
    assert features_tensor.shape[1] == 1280

    return all_labels, features_tensor


def load_model(checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = RankNetModel(input_dim=1280, hidden_dim=256, num_blocks=3, activation=nn.ReLU, dropout=0.1)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="model1",
                        help="s2s2fun model variant (model1 uses p_num=60, others use p_num=30)")
    parser.add_argument("--checkpoint", type=str, default="./mrp.pth")
    parser.add_argument("--out_file", type=str, default="./output.csv")
    parser.add_argument("--af3_out_dir", type=str, default="./af3_output")
    args = parser.parse_args()

    p_num = 60 if args.model == 'model1' else 30

    print('model:', args.model)
    print('af3_out_dir:', args.af3_out_dir)
    print('checkpoint:', args.checkpoint)

    af3_embeddings_files = get_embeddings_file(args.af3_out_dir)
    all_labels, all_features = get_feature(args.af3_out_dir, af3_embeddings_files, p_num=p_num, ligand_num=20)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('device:', device)
    print('input feature shape:', all_features.shape)

    model = load_model(args.checkpoint, device=device)
    input_tensor = all_features.detach().clone().to(device, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(input_tensor).cpu().numpy()

    df = pd.DataFrame({'seq_id': all_labels, 'score': outputs})
    df.to_csv(args.out_file, index=False)
    print('Rank score saved to:', args.out_file)


if __name__ == "__main__":
    main()
