# ============================================================
# FULL RECTIFIED EVALUATION + GRAPHING CODE
# For protein next-frame prediction project
# ============================================================

import os
import warnings
import h5py
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch_cluster import knn_graph
from torch_scatter import scatter
from e3nn import o3
from e3nn.nn import FullyConnectedNet
from e3nn.math import soft_one_hot_linspace

warnings.filterwarnings("ignore")
plt.style.use("default")

# ============================================================
# 1) Paths + device
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "preprocessed data")
BEST_PATH = os.path.join(BASE_DIR, "best_model.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print("BASE_DIR:", BASE_DIR)
print("DATA_DIR :", DATA_DIR)
print("BEST_PATH:", BEST_PATH)

if not os.path.exists(BEST_PATH):
    raise FileNotFoundError(f"Checkpoint not found: {BEST_PATH}")
if not os.path.isdir(DATA_DIR):
    raise FileNotFoundError(f"Data folder not found: {DATA_DIR}")

# ============================================================
# 2) Build sample index exactly like training
# ============================================================
def build_sample_index(data_dir):
    refs = []
    expected_nodes = None

    hdf5_files = [f for f in sorted(os.listdir(data_dir)) if f.endswith(".hdf5")]
    if not hdf5_files:
        raise FileNotFoundError(f"No .hdf5 files found in: {data_dir}")

    for file_name in hdf5_files:
        file_path = os.path.join(data_dir, file_name)
        with h5py.File(file_path, "r") as f:
            if "coordinates" not in f:
                continue

            coords = f["coordinates"]

            if coords.ndim != 3 or coords.shape[-1] != 3:
                continue
            if coords.shape[0] < 2:
                continue

            if expected_nodes is None:
                expected_nodes = coords.shape[1]

            if coords.shape[1] != expected_nodes:
                continue

            for t in range(coords.shape[0] - 1):
                refs.append((file_path, t))

    if expected_nodes is None:
        raise RuntimeError("Could not infer node count from the dataset.")

    return refs, expected_nodes


sample_refs, expected_nodes = build_sample_index(DATA_DIR)
print("Total samples:", len(sample_refs))
print("Nodes/sample :", expected_nodes)

if len(sample_refs) == 0:
    raise RuntimeError("No valid samples found after scanning the HDF5 files.")

train_refs, temp_refs = train_test_split(
    sample_refs, test_size=0.4, random_state=42, shuffle=True
)
val_refs, test_refs = train_test_split(
    temp_refs, test_size=0.5, random_state=42, shuffle=True
)

print("Train:", len(train_refs))
print("Val  :", len(val_refs))
print("Test :", len(test_refs))

# ============================================================
# 3) Dataset
# ============================================================
class ProteinDataset(Dataset):
    def __init__(self, refs):
        self.refs = refs
        self._handles = {}

    def __len__(self):
        return len(self.refs)

    def _get_handle(self, path):
        if path not in self._handles:
            self._handles[path] = h5py.File(path, "r")
        return self._handles[path]

    def __getitem__(self, idx):
        file_path, t = self.refs[idx]
        h5 = self._get_handle(file_path)
        coords = h5["coordinates"]

        x = coords[t].astype(np.float32, copy=False)
        y = coords[t + 1].astype(np.float32, copy=False)

        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
        }

    def close(self):
        for h in self._handles.values():
            try:
                h.close()
            except Exception:
                pass
        self._handles.clear()

    def __del__(self):
        self.close()


def move_batch(batch):
    return {
        "x": batch["x"].to(device, non_blocking=True),
        "y": batch["y"].to(device, non_blocking=True),
    }


def prepare_inputs(x):
    center = x.mean(dim=0, keepdim=True)
    x_rel = x - center
    radius = x_rel.norm(dim=1, keepdim=True)
    node_input = torch.cat([radius, x_rel], dim=1)
    return node_input, x_rel


NUM_WORKERS = 0
PIN_MEMORY = False

train_dataset = ProteinDataset(train_refs)
val_dataset = ProteinDataset(val_refs)
test_dataset = ProteinDataset(test_refs)

train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    collate_fn=lambda batch: batch[0],
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    collate_fn=lambda batch: batch[0],
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    collate_fn=lambda batch: batch[0],
)

print("DataLoaders ready")

# ============================================================
# 4) Build a fixed graph once
# ============================================================
with h5py.File(sample_refs[0][0], "r") as f:
    first_x = f["coordinates"][sample_refs[0][1]].astype(np.float32, copy=False)

sample_x = torch.from_numpy(first_x).float()
base_edge_index = knn_graph(sample_x, k=4, loop=False).long().contiguous()

edge_vec0 = sample_x[base_edge_index[1]] - sample_x[base_edge_index[0]]
EDGE_RADIUS_MAX = float(edge_vec0.norm(dim=1).max().item() * 1.10 + 1e-6)
AVG_NEIGHBORS = float(base_edge_index.size(1) / expected_nodes)

print("Base edge_index shape:", base_edge_index.shape)
print("Number of edges      :", base_edge_index.size(1))
print("Edge radius max      :", EDGE_RADIUS_MAX)
print("Avg neighbors        :", AVG_NEIGHBORS)

# ============================================================
# 5) Model definition (matches training script structure)
# ============================================================
class E3NNGraphConv(nn.Module):
    def __init__(self, irreps_in, irreps_out, lmax=1, num_basis=6, max_radius=10.0, avg_neighbors=4.0):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax)

        self.tp = o3.FullyConnectedTensorProduct(
            self.irreps_in,
            self.irreps_sh,
            self.irreps_out,
            shared_weights=False,
        )

        self.fc = FullyConnectedNet(
            [num_basis, 24, self.tp.weight_numel],
            torch.relu,
        )

        self.num_basis = num_basis
        self.max_radius = float(max_radius)
        self.avg_neighbors = float(avg_neighbors)

    def forward(self, x, pos, edge_index):
        edge_src, edge_dst = edge_index

        edge_vec = pos[edge_dst] - pos[edge_src]
        edge_len = edge_vec.norm(dim=1)

        sh = o3.spherical_harmonics(
            self.irreps_sh,
            edge_vec,
            normalize=True,
            normalization="component",
        )

        emb = soft_one_hot_linspace(
            edge_len,
            start=0.0,
            end=self.max_radius,
            number=self.num_basis,
            basis="smooth_finite",
            cutoff=True,
        ).mul(self.num_basis ** 0.5)

        weight = self.fc(emb).to(x.dtype)
        msg = self.tp(x[edge_src], sh, weight)

        out = scatter(msg, edge_dst, dim=0, dim_size=x.shape[0])
        out = out.div(self.avg_neighbors ** 0.5)
        return out


class E3NNDeltaPredictor(nn.Module):
    def __init__(self, edge_index, avg_neighbors, edge_radius_max):
        super().__init__()
        self.register_buffer("edge_index", edge_index, persistent=False)

        self.input_irreps = o3.Irreps("0e + 1o")
        self.hidden_irreps = o3.Irreps("8x0e + 8x1o")
        self.output_irreps = o3.Irreps("1o")

        self.input_proj = o3.Linear(self.input_irreps, self.hidden_irreps)
        self.skip = o3.Linear(self.input_irreps, self.output_irreps)

        self.conv1 = E3NNGraphConv(
            self.hidden_irreps,
            self.hidden_irreps,
            lmax=1,
            num_basis=6,
            max_radius=edge_radius_max,
            avg_neighbors=avg_neighbors,
        )
        self.conv2 = E3NNGraphConv(
            self.hidden_irreps,
            self.output_irreps,
            lmax=1,
            num_basis=6,
            max_radius=edge_radius_max,
            avg_neighbors=avg_neighbors,
        )

    def forward(self, node_input, pos):
        x = self.input_proj(node_input)
        h = self.conv1(x, pos, self.edge_index)
        out = self.conv2(h, pos, self.edge_index)
        return out + self.skip(node_input)


model = E3NNDeltaPredictor(
    edge_index=base_edge_index.to(device),
    avg_neighbors=AVG_NEIGHBORS,
    edge_radius_max=EDGE_RADIUS_MAX,
).to(device)

checkpoint = torch.load(BEST_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

history = checkpoint.get("history", {})

print("Model loaded successfully")

# ============================================================
# 6) Evaluation helpers
# ============================================================
criterion = nn.MSELoss()

@torch.inference_mode()
def evaluate_loader(model, loader):
    model.eval()

    total_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0
    total_rmsd = 0.0
    num_batches = 0

    all_atom_errors = []
    all_diff = []
    all_true = []
    all_pred = []

    for batch in loader:
        batch = move_batch(batch)
        x = batch["x"]
        y = batch["y"]

        node_input, pos = prepare_inputs(x)
        target_delta = y - x

        pred_delta = model(node_input, pos)
        pred_next = x + pred_delta

        loss = criterion(pred_delta, target_delta)
        if not torch.isfinite(loss):
            continue

        diff = pred_next - y

        total_loss += float(loss.item())
        total_mse += float((diff ** 2).mean().item())
        total_mae += float(diff.abs().mean().item())
        total_rmsd += float(torch.sqrt((diff ** 2).sum(dim=1)).mean().item())
        num_batches += 1

        all_atom_errors.append(diff.norm(dim=1).detach().cpu())
        all_diff.append(diff.detach().cpu())
        all_true.append(y.detach().cpu())
        all_pred.append(pred_next.detach().cpu())

    if num_batches == 0:
        raise RuntimeError("No valid batches were evaluated.")

    all_atom_errors = torch.cat(all_atom_errors, dim=0).numpy()
    all_diff = torch.cat(all_diff, dim=0).numpy()
    all_true = torch.cat(all_true, dim=0).numpy()
    all_pred = torch.cat(all_pred, dim=0).numpy()

    axis_mae = np.abs(all_diff).mean(axis=0)
    axis_mean_signed = all_diff.mean(axis=0)

    return {
        "loss": total_loss / num_batches,
        "mse": total_mse / num_batches,
        "mae": total_mae / num_batches,
        "rmsd": total_rmsd / num_batches,
        "atom_errors": all_atom_errors,
        "diff": all_diff,
        "true": all_true,
        "pred": all_pred,
        "axis_mae": axis_mae,
        "axis_mean_signed": axis_mean_signed,
    }


test_metrics = evaluate_loader(model, test_loader)

print("\nFINAL TEST METRICS")
print("-------------------")
print(f"Loss   : {test_metrics['loss']:.6f}")
print(f"MSE    : {test_metrics['mse']:.6f}")
print(f"MAE    : {test_metrics['mae']:.6f}")
print(f"RMSD   : {test_metrics['rmsd']:.6f}")
print(f"Axis MAE (X,Y,Z): {test_metrics['axis_mae']}")

# ============================================================
# 7) Graphs suitable for this project
# ============================================================

# --- Graph 1: Train / Val Loss
if "train_loss" in history and "val_loss" in history:
    plt.figure(figsize=(9, 4))
    plt.plot(history["train_loss"], marker="o", linewidth=2, label="Train Loss")
    plt.plot(history["val_loss"], marker="o", linewidth=2, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.close()

# --- Graph 2: MSE / MAE across epochs
if "mse" in history and "mae" in history:
    plt.figure(figsize=(9, 4))
    plt.plot(history["mse"], marker="o", linewidth=2, label="MSE")
    plt.plot(history["mae"], marker="o", linewidth=2, label="MAE")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.title("Regression Metrics Across Epochs")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.close()

# --- Graph 3: RMSD distribution on test set
plt.figure(figsize=(9, 4))
plt.hist(test_metrics["atom_errors"], bins=30, alpha=0.85)
plt.xlabel("Per-Atom Displacement Error")
plt.ylabel("Count")
plt.title("Distribution of Atomic Prediction Error")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
plt.close()

# --- Graph 4: Axis-wise MAE
plt.figure(figsize=(7, 4))
plt.bar(["X", "Y", "Z"], test_metrics["axis_mae"])
plt.xlabel("Axis")
plt.ylabel("Mean Absolute Error")
plt.title("Axis-wise Prediction Error")
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.show()
plt.close()

# --- Graph 5: Mean signed error by axis
plt.figure(figsize=(7, 4))
plt.bar(["X", "Y", "Z"], test_metrics["axis_mean_signed"])
plt.xlabel("Axis")
plt.ylabel("Mean Signed Error")
plt.title("Mean Signed Prediction Error by Axis")
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.show()
plt.close()

# --- Graph 6: Predicted vs actual scatter (sampled)
flat_true = test_metrics["true"].reshape(-1, 3)
flat_pred = test_metrics["pred"].reshape(-1, 3)

sample_size = min(5000, len(flat_true))
idx = np.random.choice(len(flat_true), size=sample_size, replace=False)

plt.figure(figsize=(6, 6))
plt.scatter(flat_true[idx, 0], flat_pred[idx, 0], s=8, alpha=0.35, label="X")
plt.scatter(flat_true[idx, 1], flat_pred[idx, 1], s=8, alpha=0.35, label="Y")
plt.scatter(flat_true[idx, 2], flat_pred[idx, 2], s=8, alpha=0.35, label="Z")
min_v = min(flat_true[idx].min(), flat_pred[idx].min())
max_v = max(flat_true[idx].max(), flat_pred[idx].max())
plt.plot([min_v, max_v], [min_v, max_v], "k--", linewidth=1)
plt.xlabel("Actual Coordinate Value")
plt.ylabel("Predicted Coordinate Value")
plt.title("Predicted vs Actual Coordinates")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
plt.close()

# --- Graph 7: Histogram of per-batch RMSD-like values
# Compute per-sample RMSD more explicitly from saved arrays
diff = test_metrics["diff"]
per_atom_rms = np.sqrt((diff ** 2).sum(axis=1))  # one value per atom
plt.figure(figsize=(9, 4))
plt.hist(per_atom_rms, bins=30, alpha=0.85)
plt.xlabel("Per-Atom RMSD")
plt.ylabel("Count")
plt.title("Per-Atom RMSD Distribution")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
plt.close()

# ============================================================
# 8) Optional: print a compact summary for report use
# ============================================================
print("\nSUMMARY")
print("-------")
print(f"Train Loss last: {history['train_loss'][-1]:.6f}" if "train_loss" in history and len(history["train_loss"]) else "Train history unavailable")
print(f"Val Loss last  : {history['val_loss'][-1]:.6f}" if "val_loss" in history and len(history["val_loss"]) else "Val history unavailable")
print(f"Test Loss      : {test_metrics['loss']:.6f}")
print(f"Test MSE       : {test_metrics['mse']:.6f}")
print(f"Test MAE       : {test_metrics['mae']:.6f}")
print(f"Test RMSD      : {test_metrics['rmsd']:.6f}")

# ============================================================
# 9) Cleanup
# ============================================================
train_dataset.close()
val_dataset.close()
test_dataset.close()