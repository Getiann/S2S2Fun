# S2S2Fun: Decoding Protein Function from Latent Structural Representations

**Under review at the GEM Workshop, ICLR 2026**

S2S2Fun predicts and ranks protein function from AlphaFold3 latent structural representations. By extracting pair and single embeddings from AF3's Pairformer module and training a pairwise ranking model (RankNet), S2S2Fun captures ligand-aware structural features that are sensitive to sequence variation — without relying on evolutionary fitness proxies from protein language models.

A domain-adversarial variant (**S2S2Fun-DA**) extends this to proteins from structurally distinct families that share the same function, using a Gradient Reversal Layer to learn fold-invariant representations.

---

## Results

| Dataset | Method | Pair Accuracy | Spearman | Pearson |
|---|---|---|---|---|
| MRP (retinal-binding) | ESM3 | .745 | .645 | .405 |
| MRP | **S2S2Fun** | **.772** | **.669** | **.865** |
| GFP-like | ESM3 | .781 | .730 | .643 |
| GFP-like | **S2S2Fun** | **.831** | **.846** | **.813** |
| MRP→CRBP transfer | S2S2Fun | .717 | .615 | .569 |
| MRP→CRBP transfer | **S2S2Fun-DA** | **.775** | **.747** | **.717** |

---

## Installation

```bash
conda env create -f environment.yaml
conda activate absdesign
```

---

## Quick Start

### Inference with pretrained models

Download pretrained checkpoints:
- `mrp.pth` — S2S2Fun trained on MRP absorption peaks
- `mrp_s2s2fun_da.pt` — S2S2Fun-DA trained on MRP with CRBP transfer

```bash
# S2S2Fun inference
python inference_s2s2fun.py \
    --af3_out_dir ./af3_output \
    --checkpoint ./mrp.pth \
    --out_file ./predictions.csv

# S2S2Fun-DA inference
python inference_s2s2fun_da.py \
    --af3_out_dir ./af3_output \
    --checkpoint ./mrp_s2s2fun_da.pt \
    --out_file ./predictions_da.csv
```

### Training on your own data

See **`tutorial.ipynb`** for a step-by-step walkthrough, including:
- How to format your AF3 output directory
- How to prepare the data CSV
- Training S2S2Fun and S2S2Fun-DA from scratch
- Evaluating predictions

```bash
# Train S2S2Fun
python train_s2s2fun.py \
    --af3_out_dir ./af3_output \
    --train_data_info ./data.csv \
    --epochs 3000 \
    --save ./model.pth

# Train S2S2Fun-DA (domain adversarial)
python train_s2s2fun_da.py \
    --af3_out_dir ./af3_output \
    --data_info ./data.csv \
    --num_epochs 5000 \
    --loss_type pairwise \
    --DANN
```

---

## AF3 Input Requirements

Run AlphaFold3 on each protein–ligand complex with `save_embeddings=true`. Required settings from the paper:

- 1 random seed, 5 sampling runs, 10 recycling iterations
- MSA enabled (except HBI dataset: no MSA)
- For retinal-binding proteins: set Schiff-base bond (NZ of Lys ↔ N15 of retinal); remove retinal oxygen atom

AF3 output directory layout:
```
af3_output/
├── proteinA/
│   ├── proteinA_ranking_scores.csv
│   ├── proteinA_model.pdb
│   └── seed-1_embeddings/
│       └── proteinA_embeddings.npz
```

---

## Data CSV Format

### S2S2Fun
| column | description |
|---|---|
| `seq_id` | must match AF3 output folder name (case-insensitive) |
| `absorbance` | measured functional value |
| `set` | `train` or `test` |
| `crbp` | `False` (set to False for all when not using DA) |
| `k_pos` | key residue position (any non-NaN value) |

### S2S2Fun-DA (additional columns)
| column | description |
|---|---|
| `crbp` | `False` = source domain · `True` = target domain |

---

## Code Structure

```
S2S2Fun/
├── models.py                  # RankNetModel, FeatureExtractor, DomainClassifier, GRL
├── features.py                # AF3 feature extraction utilities
├── metric.py                  # Pair accuracy, NDCG, Pearson, Spearman, AP
├── train_s2s2fun.py           # S2S2Fun training script
├── train_s2s2fun_da.py        # S2S2Fun-DA training script
├── inference_s2s2fun.py       # Inference with S2S2Fun checkpoint
├── inference_s2s2fun_da.py    # Inference with S2S2Fun-DA checkpoint
├── tutorial.ipynb             # Step-by-step training tutorial
├── data/
│   ├── final_FP.csv           # GFP-like fluorescent protein dataset
│   ├── HBI_binding.csv        # HBI brightness dataset
│   └── HBI_stability.csv      # HBI stability dataset
└── environment.yaml           # Conda environment
```

---

## Feature Extraction

For a protein with `L = Nprotein + Nligand` tokens, S2S2Fun extracts:

1. **Nearest-residue cropping** — selects the `Nprotein=30` residues closest to the ligand binding site from the AF3 structure
2. **Block-wise mean pooling** over the pair representation (PP, LL, PL, LP blocks → 512-dim)
3. **Mean pooling** over protein and ligand single representations (768-dim)
4. Concatenation → **1280-dim feature vector**

---

## Citation

```bibtex
@inproceedings{s2s2fun2026,
  title     = {S2S2Fun: Decoding Protein Function from Latent Structural Representations},
  booktitle = {GEM Workshop, ICLR 2026},
  year      = {2026},
  note      = {Under review}
}
```
