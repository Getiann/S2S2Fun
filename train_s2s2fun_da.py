import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
import pandas as pd
import glob
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.utils.tensorboard import SummaryWriter

from models import RankNetModel, FeatureExtractor, DomainClassifier, GRL
from features import (
    cached_extract_sequences, extract_features, get_embeddings_file
)
from metric import evaluate


def MSELoss(pred, target):
    return nn.MSELoss()(pred, target)


def ListNetLoss(pred, target):
    pred_log_softmax = F.log_softmax(pred, dim=0)
    target_softmax = F.softmax(target, dim=0)
    loss = -torch.sum(target_softmax * pred_log_softmax)
    return loss / target.size(0)


def CalRankNetLoss(pred_i, pred_j, target, y_i, y_j, y0, sigmoid_scale=1.0):
    diff = pred_i - pred_j
    rank_loss = nn.BCEWithLogitsLoss()(diff * sigmoid_scale, target)

    pred = torch.cat([pred_i, pred_j], dim=0) * sigmoid_scale
    labels = torch.cat([y_i, y_j], dim=0)
    mask_higher = labels > y0
    mask_lower = labels < y0

    loss_higher = -torch.log(torch.sigmoid(pred[mask_higher]))
    loss_lower = -torch.log(torch.sigmoid(-pred[mask_lower]))
    logloss = (loss_higher.sum() + loss_lower.sum()) / labels.size(0)

    return rank_loss + logloss


def CalListNetLoss(pred, labels, y0):
    loss = -torch.sum(labels * pred) + (y0 + torch.sum(labels)) * torch.log(1 + torch.sum(torch.exp(pred)))
    return loss / labels.size(0)


class RankNetLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_i, pred_j, target):
        return nn.BCEWithLogitsLoss()(pred_i - pred_j, target)


class PairDataset(Dataset):
    def __init__(self, embeddings, labels, mode='random', pairs=None, pairs_per_epoch=10000, skip_equal=True):
        self.x = torch.as_tensor(embeddings, dtype=torch.float32)
        self.y = labels
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
        return self.x[i], self.x[j], torch.tensor(target, dtype=torch.float32), i, j, self.y[i], self.y[j]


class DomainDataset(Dataset):
    def __init__(self, pair, y, domain_label, name):
        assert len(pair) == len(y), "Length mismatch"
        self.p = torch.as_tensor(np.asarray(pair), dtype=torch.float32)
        self.y = torch.as_tensor(np.asarray(y), dtype=torch.float32)
        self.domain_label = torch.as_tensor(np.asarray(domain_label), dtype=torch.float32)
        self.name = name
        self.n = len(self.p)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.p[idx], self.y[idx], self.domain_label[idx], self.name[idx]


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

        feature = extract_features(pair, single, N_protein=len(seqs), N_ligand=ligand_num)
        return seq_name, feature
    else:
        return seq_name, torch.from_numpy(pair), torch.from_numpy(single)


def get_feature(af3_out_dir, af3_embeddings_files, p_num=30, ligand_num=20, diag=False, pooling=True):
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


def train_dann(Gf, Gc, Gd, stage1_source_loader, stage2_source_loader, target_loader,
               num_epochs=50, lambda_max=1.0, stage1_epoch=100, device="cuda",
               tb_folder='./tb_logs', target_dataset=None, test_dataset=None,
               train_name='train', ck_epoch=400, loss_type='pointwise',
               a=0.5, y0=0.0, cal=True, sigmoid_scale=1.0, DANN=True,
               checkpoint_dir='./checkpoint', out_csv_dir='./csv'):

    writer = SummaryWriter(log_dir=tb_folder)
    Gf, Gc, Gd = Gf.to(device), Gc.to(device), Gd.to(device)
    grl = GRL(lambda_=0.0)

    if DANN:
        optimizer = torch.optim.Adam(
            list(Gf.parameters()) + list(Gc.parameters()) + list(Gd.parameters()), lr=1e-4
        )
    else:
        optimizer = torch.optim.Adam(list(Gc.parameters()), lr=1e-4)

    bce_loss = nn.BCEWithLogitsLoss()
    rank_loss_fn = RankNetLoss()
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(out_csv_dir, exist_ok=True)
    evalue_df = pd.DataFrame()

    for epoch in range(num_epochs):
        Gf.train(); Gc.train(); Gd.train()
        target_iter = iter(target_loader)

        run_task_loss = 0.0
        run_total_loss = 0.0
        n_seen = 0
        source_loader = stage1_source_loader if epoch < stage1_epoch else stage2_source_loader

        for (pair_s_batch, y_batch, source_domain_label, source_name) in source_loader:
            try:
                pair_t_batch, _, _, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                pair_t_batch, _, _, _ = next(target_iter)

            pair_s_batch = pair_s_batch.to(device)
            pair_t_batch = pair_t_batch.to(device)
            y_batch = y_batch.to(device)

            feat = Gf(pair_s_batch) if DANN else pair_s_batch
            out = Gc(feat)
            mse_loss = MSELoss(out, y_batch)

            if loss_type == 'pointwise':
                loss_task = mse_loss
            elif loss_type == 'pairwise':
                pairs_dataset = PairDataset(feat, y_batch, mode='pairs', skip_equal=True)
                pairs_loader = DataLoader(pairs_dataset, batch_size=len(pairs_dataset), shuffle=False, num_workers=0)
                input_i, input_j, target_rank, _, _, y_i, y_j = next(iter(pairs_loader))
                input_i = input_i.to(device)
                input_j = input_j.to(device)
                target_rank = target_rank.to(device)
                y_i = y_i.to(device)
                y_j = y_j.to(device)
                out_i = Gc(input_i)
                out_j = Gc(input_j)
                if cal:
                    rank_loss = CalRankNetLoss(out_i, out_j, target_rank, y_i, y_j, y0, sigmoid_scale)
                    assert a == 1.0, "When cal is True, a must be 1.0"
                else:
                    rank_loss = rank_loss_fn(out_i, out_j, target_rank)
                loss_task = a * rank_loss + (1 - a) * mse_loss
            elif loss_type == 'listwise':
                list_loss = ListNetLoss(out, y_batch)
                loss_task = a * list_loss + (1 - a) * mse_loss
            else:
                raise ValueError("loss_type must be 'pointwise', 'pairwise' or 'listwise'")

            if DANN:
                feat_s = feat.detach()
                feat_t = Gf(pair_t_batch)
                feat_concat = torch.cat([feat_s, feat_t], dim=0)

                p = epoch / num_epochs
                lambda_ = lambda_max * (2 / (1 + np.exp(-10 * p)) - 1)
                grl.lambda_ = lambda_

                domain_pred = Gd(grl(feat_concat)).squeeze(-1)
                domain_labels = torch.cat([
                    source_domain_label,
                    torch.ones(feat_t.size(0))
                ]).to(device)
                loss_domain = bce_loss(domain_pred, domain_labels)
                loss = loss_task + lambda_ * loss_domain
            else:
                loss = loss_task

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            run_task_loss += loss_task.item()
            run_total_loss += loss.item()
            n_seen += 1

        if epoch > ck_epoch and epoch % 5 == 0:
            torch.save({
                'Gf_state_dict': Gf.state_dict(),
                'Gc_state_dict': Gc.state_dict(),
                'Gd_state_dict': Gd.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
            }, os.path.join(checkpoint_dir, f'{train_name}_epoch_{epoch}.pt'))

        with torch.no_grad():
            Gf.eval(); Gc.eval(); Gd.eval()
            target_loader_eval = DataLoader(target_dataset, batch_size=len(target_dataset), shuffle=False)
            pair_t_batch, y_t_batch, target_domain_label, target_name = next(iter(target_loader_eval))
            pair_t_batch = pair_t_batch.to(device)
            feat_t = Gf(pair_t_batch)
            scores_t = Gc(feat_t)

            ifpointwise = loss_type == 'pointwise'
            target_e = evaluate(scores_t.cpu().numpy(), y_t_batch.numpy(), k=None, skip_equal=True, pointwise=ifpointwise)

            accu = target_e['pairwise_accuracy']

            if test_dataset is not None:
                test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
                pair_test_batch, y_test_batch, valid_domain_label, test_name = next(iter(test_loader))
                pair_test_batch = pair_test_batch.to(device)
                valid_domain_label = valid_domain_label.to(device)
                feat_test = Gf(pair_test_batch)
                scores_test = Gc(feat_test)
                y_test_batch = y_test_batch.to(device)

                br_score = scores_test[valid_domain_label == 1]
                br_y = y_test_batch[valid_domain_label == 1]
                crbp_score = scores_test[valid_domain_label == 0]
                crbp_y = y_test_batch[valid_domain_label == 0]

                br_e = evaluate(br_score.cpu().numpy(), br_y.cpu().numpy(), k=None, skip_equal=True, pointwise=False)
                crbp_e = evaluate(crbp_score.cpu().numpy(), crbp_y.cpu().numpy(), k=None, skip_equal=True, pointwise=False)

                accu = br_e['pairwise_accuracy']
                new_row = pd.DataFrame({
                    'epoch': epoch,
                    'test_pred': ",".join(map(str, scores_test.cpu().numpy())),
                    'test_true': ",".join(map(str, y_test_batch.cpu().numpy())),
                    'test_name': ",".join(test_name),
                    'target_pred': ",".join(map(str, scores_t.cpu().numpy())),
                    'target_true': ",".join(map(str, y_t_batch.numpy())),
                    'target_name': ",".join(target_name),
                    'crbp_pairwise_accuracy': target_e['pairwise_accuracy'],
                    'crbp_ndcg': target_e['ndcg'],
                    'crbp_pearson_correlation': target_e['pearson_correlation'],
                    'crbp_spearman_correlation': target_e['spearman_correlation'],
                    'crbp_AP_by_threshold': target_e['AP_by_threshold'],
                    'onlycrbp_pairwise_accuracy': crbp_e['pairwise_accuracy'],
                    'onlycrbp_ndcg': crbp_e['ndcg'],
                    'onlycrbp_pearson_correlation': crbp_e['pearson_correlation'],
                    'onlycrbp_spearman_correlation': crbp_e['spearman_correlation'],
                    'onlycrbp_AP_by_threshold': crbp_e['AP_by_threshold'],
                    'mr_pairwise_accuracy': br_e['pairwise_accuracy'],
                    'mr_ndcg': br_e['ndcg'],
                    'mr_pearson_correlation': br_e['pearson_correlation'],
                    'mr_spearman_correlation': br_e['spearman_correlation'],
                    'mr_AP_by_threshold': br_e['AP_by_threshold'],
                }, index=[0])
                evalue_df = pd.concat([evalue_df, new_row], ignore_index=True)
                evalue_df.to_csv(os.path.join(out_csv_dir, f'evaluation_{train_name}.csv'), index=False)

        writer.add_scalar(f'{train_name}/loss_Task', run_task_loss / n_seen, epoch)
        writer.add_scalar(f'{train_name}/loss_Total', run_total_loss / n_seen, epoch)
        writer.add_scalar(f'{train_name}/accu_Target', accu, epoch)

        print(f"Epoch [{epoch + 1}/{num_epochs}] | "
              f"Task Loss: {run_task_loss / n_seen:.4f} | "
              f"Target Test accu: {accu:.4f}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--af3_out_dir', type=str, required=True)
    parser.add_argument('--data_info', type=str, required=True)
    parser.add_argument('--num_epochs', type=int, default=5000)
    parser.add_argument('--tb_folder', type=str, default='./tb_logs')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint')
    parser.add_argument('--out_csv_dir', type=str, default='./csv')
    parser.add_argument('--train_name', type=str, default='s2s2fun_da')
    parser.add_argument('--loss_type', type=str, default='pointwise',
                        choices=['pointwise', 'pairwise', 'listwise'])
    parser.add_argument('--lambda_max', type=float, default=0.0)
    parser.add_argument('--stage1_epoch', type=int, default=20000)
    parser.add_argument('--ck_epoch', type=int, default=10000)
    parser.add_argument('--DANN', action='store_true', default=False)
    args = parser.parse_args()

    p_num = 30
    input_dim = 1280

    data_info = pd.read_csv(args.data_info)
    data_info = data_info[~data_info['seq_id'].str.contains('design')]
    data_info = data_info[data_info['k_pos'].notna()]

    all_seq_ids = [s.lower() for s in data_info['seq_id'].tolist()]
    af3_embeddings_files = get_embeddings_file(args.af3_out_dir, all_seq_ids)

    assert len(af3_embeddings_files) == len(data_info), "Embeddings files and data info length mismatch"
    all_labels, all_features = get_feature(args.af3_out_dir, af3_embeddings_files, p_num=p_num, ligand_num=20, diag=False, pooling=True)

    y_abs = np.array([
        data_info.loc[data_info['seq_id'].str.lower() == label.lower(), 'absorbance'].values[0]
        for label in all_labels
    ])
    y0 = np.mean(y_abs)
    print("y0:", y0)

    data_info = data_info.copy()
    data_info['seq_id_lower'] = data_info['seq_id'].str.lower()
    all_labels = [s.lower() for s in all_labels]

    df_map = data_info.set_index('seq_id_lower').loc[all_labels, ['crbp', 'set']]
    all_names = data_info.set_index('seq_id_lower').loc[all_labels, 'seq_id'].tolist()

    domain_label = np.array((~df_map['crbp']).astype(int).tolist())
    domain_mask = np.array((df_map['crbp']).astype(bool).tolist())

    stage1_source_train_mask = np.array((~df_map['crbp']) & (df_map['set'] == 'train')).astype(bool).tolist()
    stage2_source_train_mask = np.array((df_map['set'] == 'train').astype(bool).tolist())
    source_test_mask = np.array((df_map['set'] == 'test')).astype(bool).tolist()

    stage1_source_dataset = DomainDataset(
        all_features[stage1_source_train_mask],
        y_abs[stage1_source_train_mask],
        domain_label[stage1_source_train_mask].astype(float),
        np.array(all_names)[stage1_source_train_mask].tolist()
    )
    stage2_source_dataset = DomainDataset(
        all_features[stage2_source_train_mask],
        y_abs[stage2_source_train_mask],
        domain_label[stage2_source_train_mask].astype(float),
        np.array(all_names)[stage2_source_train_mask].tolist()
    )
    target_dataset = DomainDataset(
        all_features[domain_mask],
        y_abs[domain_mask],
        domain_label[domain_mask].astype(float),
        np.array(all_names)[domain_mask].tolist()
    )
    source_dataset_test = DomainDataset(
        all_features[source_test_mask],
        y_abs[source_test_mask],
        domain_label[source_test_mask].astype(float),
        np.array(all_names)[source_test_mask].tolist()
    )

    stage1_source_loader = DataLoader(stage1_source_dataset, batch_size=256, shuffle=False)
    stage2_source_loader = DataLoader(stage2_source_dataset, batch_size=1028, shuffle=False)
    target_loader = DataLoader(target_dataset, batch_size=1028, shuffle=False)

    Gf = FeatureExtractor(dim=input_dim, hidden=512, dropout=0.1)
    Gc = RankNetModel(input_dim=input_dim)
    Gd = DomainClassifier(input_dim=input_dim)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dann(
        Gf, Gc, Gd,
        stage1_source_loader, stage2_source_loader, target_loader,
        num_epochs=args.num_epochs, lambda_max=args.lambda_max,
        stage1_epoch=args.stage1_epoch, device=device,
        tb_folder=args.tb_folder, target_dataset=target_dataset,
        test_dataset=source_dataset_test, train_name=args.train_name,
        ck_epoch=args.ck_epoch, loss_type=args.loss_type,
        a=1.0, y0=y0, cal=False, sigmoid_scale=1.0, DANN=args.DANN,
        checkpoint_dir=args.checkpoint_dir, out_csv_dir=args.out_csv_dir
    )


if __name__ == "__main__":
    main()
