import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import os

from models import RankNetModel
from features import (
    cached_extract_sequences, cached_get_nearest_residues,
    extract_features, get_embeddings_file
)
from metric import evaluate


def MSELoss(pred, target):
    return nn.MSELoss()(pred, target)


def ListNetLoss(pred, target):
    pred_log_softmax = F.log_softmax(pred, dim=0)
    target_softmax = F.softmax(target, dim=0)
    loss = -torch.sum(target_softmax * pred_log_softmax)
    return loss / target.size(0)


def _process_single_file(args):
    embeddings_file, af3_out_dir, p_num, ligand_num, diag, pooling = args
    seq_name = embeddings_file.split('/')[-3]

    data = np.load(embeddings_file, mmap_mode='r')
    pair = data['pair_embeddings']
    single = data['single_embeddings']

    if not diag:
        for i in range(pair.shape[2]):
            np.fill_diagonal(pair[:, :, i], 0.0)

    if pooling:
        pdb_file = os.path.join(af3_out_dir, f"{seq_name.lower()}/{seq_name.lower()}_model.pdb")
        seqs = cached_extract_sequences(pdb_file)['A']
        all_tokens = len(seqs) + ligand_num

        if pair.shape[0] == all_tokens + 1:
            ligand_index = list(range(all_tokens - ligand_num, all_tokens + 1))
            ligand_index.remove(all_tokens - 5)
        else:
            ligand_index = list(range(all_tokens - ligand_num, all_tokens))
            assert pair.shape[0] == all_tokens, \
                f"pair shape {pair.shape} does not match sequence length {len(seqs)} + {ligand_num}"

        near_index = [i - 1 for i in sorted(cached_get_nearest_residues(pdb_file, "RET", p_num))]
        index_np = np.array(near_index + ligand_index)
        feature = extract_features(
            torch.from_numpy(pair[index_np][:, index_np, :]),
            torch.from_numpy(single[index_np, :]),
            N_protein=p_num, N_ligand=ligand_num
        )
        return seq_name, feature
    else:
        return seq_name, torch.from_numpy(pair), torch.from_numpy(single)


def get_feature(af3_out_dir, af3_embeddings_files, p_num=30, ligand_num=20, diag=True, pooling=True):
    n_files = len(af3_embeddings_files)
    feature_tensor = torch.empty((n_files, 1280), dtype=torch.float32)
    all_labels = [None] * n_files

    args_list = [(f, af3_out_dir, p_num, ligand_num, diag, pooling) for f in af3_embeddings_files]
    with ThreadPoolExecutor(max_workers=64) as executor:
        futures = {executor.submit(_process_single_file, args): idx for idx, args in enumerate(args_list)}
        for future in as_completed(futures):
            idx = futures[future]
            seq_name, feature = future.result()
            all_labels[idx] = seq_name
            feature_tensor[idx] = feature

    return all_labels, feature_tensor


class PairDataset(Dataset):
    def __init__(self, embeddings, labels, mode='random', pairs=None, pairs_per_epoch=10000, skip_equal=True):
        self.x = torch.as_tensor(np.asarray(embeddings), dtype=torch.float32)
        self.y = np.asarray(labels)
        self.n = len(self.y)
        self.skip_equal = skip_equal
        self.mode = mode
        if mode == 'pairs':
            if pairs is None:
                pairs = [
                    (i, j)
                    for i in range(self.n)
                    for j in range(self.n)
                    if i != j and not (skip_equal and self.y[i] == self.y[j])
                ]
            self.pairs = pairs
            self.length = len(self.pairs)
        elif mode == 'random':
            self.pairs_per_epoch = int(pairs_per_epoch)
            self.length = self.pairs_per_epoch
        else:
            raise ValueError("mode must be 'random' or 'pairs'")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.mode == 'pairs':
            i, j = self.pairs[idx]
        else:
            while True:
                i = random.randrange(self.n)
                j = random.randrange(self.n)
                if i == j:
                    continue
                if self.skip_equal and self.y[i] == self.y[j]:
                    continue
                break
        yi, yj = self.y[i], self.y[j]
        if yi > yj:
            target = 1.0
        elif yi < yj:
            target = 0.0
        else:
            target = 0.5
        return self.x[i], self.x[j], torch.tensor(target, dtype=torch.float32), self.y[i], self.y[j]


def train(model, embeddings, labels, epochs=5, batch_size=256, lr=1e-3,
          pairs_per_epoch=5000, mode='random', device=None,
          val_data=None, skip_equal=True, save_path=None, log_every=1,
          tb_folder='tb_logs', name='train', out_csv_dir='.'):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    writer = SummaryWriter(log_dir=tb_folder)
    dataset = PairDataset(embeddings, labels, mode=mode, pairs_per_epoch=pairs_per_epoch, skip_equal=skip_equal)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    os.makedirs(out_csv_dir, exist_ok=True)
    evalue_df = pd.DataFrame()
    history = {}
    e_results = None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_seen = 0
        for xb, xj, tgt, yi, yj in loader:
            xb, xj, tgt = xb.to(device), xj.to(device), tgt.to(device)
            yi, yj = yi.to(device), yj.to(device)
            si = model(xb)
            sj = model(xj)
            loss = criterion(si - sj, tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            bs = xb.size(0)
            running_loss += loss.item() * bs
            n_seen += bs

        avg_loss = running_loss / max(1, n_seen)

        if val_data is not None:
            model.eval()
            with torch.no_grad():
                val_emb, val_labels, all_seqid = val_data
                val_emb_t = torch.as_tensor(val_emb, dtype=torch.float32).to(device)
                scores = model(val_emb_t).cpu().numpy()
                e_results = evaluate(val_labels.cpu().numpy(), scores, k=None, skip_equal=skip_equal)

            new_row = pd.DataFrame({
                'epoch': epoch,
                'pred': ",".join(scores.astype(str)),
                'true': ",".join(val_labels.cpu().numpy().astype(str)),
                'seq_id': ",".join(all_seqid),
                'pairwise_accuracy': e_results['pairwise_accuracy'],
                'ndcg': e_results['ndcg'],
                'pearson_correlation': e_results['pearson_correlation'],
                'spearman_correlation': e_results['spearman_correlation'],
                'AP_by_threshold': e_results['AP_by_threshold'],
            }, index=[0])
            evalue_df = pd.concat([evalue_df, new_row], ignore_index=True)
            evalue_df.to_csv(os.path.join(out_csv_dir, f'evaluation_{name}.csv'), index=False)

        if epoch % log_every == 0:
            val_acc_str = f" val_pair_acc={e_results['pairwise_accuracy']:.4f}" if e_results else ""
            print(f"[Epoch {epoch}/{epochs}] loss={avg_loss:.6f}{val_acc_str}")

        history[epoch] = dict(loss=avg_loss, val_pair_acc=e_results['pairwise_accuracy'] if e_results else None)
        writer.add_scalar(f'ranknet_{name}_{batch_size}/train', avg_loss, epoch)
        if e_results:
            writer.add_scalar(f'ranknet_{name}_{batch_size}/val_pair_acc', e_results['pairwise_accuracy'], epoch)
        writer.add_scalar(f'ranknet_{name}_{batch_size}/learning_rate', opt.param_groups[0]['lr'], epoch)

        if save_path is not None and epoch >= 760:
            save_path_epoch = save_path.replace('.pth', f'_{epoch}.pth')
            torch.save({'model_state': model.state_dict(), 'epoch': epoch, 'history': history}, save_path_epoch)

    return model, history


def extract_features_from_af3dir(af3_out_dir, data_info):
    p_num = 30
    all_seq_ids = [s.lower() for s in data_info['seq_id'].tolist()]
    af3_embeddings_files = get_embeddings_file(af3_out_dir, all_seq_ids)

    assert len(af3_embeddings_files) == len(data_info), "Embeddings files and data info length mismatch"
    all_labels, all_features = get_feature(af3_out_dir, af3_embeddings_files, p_num=p_num, ligand_num=20, diag=True, pooling=True)

    y_abs = np.array([
        data_info.loc[data_info['seq_id'].str.lower() == label.lower(), 'absorbance'].values[0]
        for label in all_labels
    ])
    return all_features, all_labels, y_abs


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--pairs_per_epoch', type=int, default=4000)
    parser.add_argument('--hidden', type=str, default='256')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--save', type=str, default=None)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--af3_out_dir', type=str, required=True)
    parser.add_argument('--train_data_info', type=str, required=True)
    parser.add_argument('--input_dim', type=int, default=1280)
    parser.add_argument('--tb_folder', type=str, default='./tb_logs')
    parser.add_argument('--out_csv_dir', type=str, default='./csv')
    parser.add_argument('--train_name', type=str, default='s2s2fun')
    args = parser.parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_info = pd.read_csv(args.train_data_info)
    data_info = data_info[data_info['crbp'] == False]
    data_info = data_info[~data_info['seq_id'].str.contains('design')]
    data_info = data_info[data_info['k_pos'].notna()]

    all_features, all_labels, y_abs = extract_features_from_af3dir(args.af3_out_dir, data_info=data_info)
    all_labels = [s.lower() for s in all_labels]
    data_info['seq_id_lower'] = data_info['seq_id'].str.lower()
    data_info = data_info.set_index('seq_id_lower').loc[all_labels, :]

    train_mask = data_info['set'] == 'train'
    test_mask = data_info['set'] == 'test'

    X_train = all_features[train_mask.values]
    y_train = torch.tensor(y_abs[train_mask.values], dtype=torch.float32)
    X_val = all_features[test_mask.values]
    y_val = torch.tensor(y_abs[test_mask.values], dtype=torch.float32)
    all_seqid = np.array(all_labels)[test_mask.values]

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)

    model = RankNetModel(input_dim=args.input_dim, hidden_dim=int(args.hidden), num_blocks=3, dropout=args.dropout)
    model, history = train(
        model, X_train, y_train,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        pairs_per_epoch=args.pairs_per_epoch, mode='pairs',
        val_data=(X_val, y_val, all_seqid), skip_equal=True,
        save_path=args.save, log_every=1,
        tb_folder=args.tb_folder, name=args.train_name,
        out_csv_dir=args.out_csv_dir
    )
    print("Training done. Saved checkpoint to", args.save)


if __name__ == '__main__':
    main()
