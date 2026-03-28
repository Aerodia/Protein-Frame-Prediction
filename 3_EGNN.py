import os
import warnings
import h5py
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from torch_cluster import knn_graph
from torch_scatter import scatter
from e3nn import o3
from e3nn.nn import FullyConnectedNet
from e3nn.math import soft_one_hot_linspace

warnings.filterwarnings("ignore")

# ============================================================
# Runtime setup
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "preprocessed data")
BEST_PATH = os.path.join(BASE_DIR, "best_model.pth")
LAST_PATH = os.path.join(BASE_DIR, "last_model.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🚀 Device:", device)

torch.set_num_threads(min(8, os.cpu_count() or 8))
try:
    torch.set_num_interop_threads(1)
except Exception:
    pass

if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

# ============================================================
# Fast HDF5 index build
# 60:20:20 split
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
print("✅ Total samples:", len(sample_refs))
print("✅ Nodes per sample:", expected_nodes)

if len(sample_refs) == 0:
    raise RuntimeError("No valid samples found after scanning the HDF5 files.")

train_refs, temp_refs = train_test_split(
    sample_refs, test_size=0.4, random_state=42, shuffle=True
)
val_refs, test_refs = train_test_split(
    temp_refs, test_size=0.5, random_state=42, shuffle=True
)

print("Train:", len(train_refs))
print("Val:  ", len(val_refs))
print("Test: ", len(test_refs))

# ============================================================
# Build a fixed graph once
# Faster graph: k=4 instead of k=8
# ============================================================
with h5py.File(sample_refs[0][0], "r") as f:
    first_x = f["coordinates"][sample_refs[0][1]].astype(np.float32, copy=False)

sample_x = torch.from_numpy(first_x).float()
base_edge_index = knn_graph(sample_x, k=4, loop=False).long().contiguous()

edge_vec0 = sample_x[base_edge_index[1]] - sample_x[base_edge_index[0]]
EDGE_RADIUS_MAX = float(edge_vec0.norm(dim=1).max().item() * 1.10 + 1e-6)
AVG_NEIGHBORS = float(base_edge_index.size(1) / expected_nodes)

print("✅ Base edge_index shape:", base_edge_index.shape)
print("✅ Number of edges:", base_edge_index.size(1))
print("✅ Edge radius max:", EDGE_RADIUS_MAX)
print("✅ Avg neighbors:", AVG_NEIGHBORS)

# ============================================================
# Dataset with open-file cache
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

print("✅ DataLoaders ready")

# ============================================================
# E3NN graph convolution blocks
# Smaller irreps + fewer layers = faster
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

# ============================================================
# Train setup
# ============================================================
model = E3NNDeltaPredictor(
    edge_index=base_edge_index.to(device),
    avg_neighbors=AVG_NEIGHBORS,
    edge_radius_max=EDGE_RADIUS_MAX,
).to(device)

# Optional compile if available
try:
    model = torch.compile(model)
    print("✅ torch.compile enabled")
except Exception:
    print("ℹ️ torch.compile not enabled")

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

print("✅ Model initialized")
print(model)

# ============================================================
# Helpers
# ============================================================
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


def find_best_threshold(labels, scores):
    thresholds = np.linspace(0.1, 0.9, 50)
    best_thresh = 0.5
    best_f1 = 0.0

    for t in thresholds:
        preds = (scores > t).astype(np.int32)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(t)

    return best_thresh, float(best_f1)


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return True
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

        return False

# ============================================================
# Sanity check
# ============================================================
batch = move_batch(next(iter(train_loader)))
node_input, pos = prepare_inputs(batch["x"])

print("x:", batch["x"].shape)
print("y:", batch["y"].shape)
print("node_input:", node_input.shape)
print("pos:", pos.shape)
print("edge_index:", model.edge_index.shape)

assert batch["x"].shape == batch["y"].shape
assert model.edge_index.shape[0] == 2
print("✅ Everything aligned")

# ============================================================
# Training / evaluation
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc=f"🟢 Train Epoch {epoch}", leave=False)

    for batch_idx, batch in enumerate(pbar):
        batch = move_batch(batch)
        x = batch["x"]
        y = batch["y"]

        node_input, pos = prepare_inputs(x)
        target_delta = y - x

        optimizer.zero_grad(set_to_none=True)

        pred_delta = model(node_input, pos)
        loss = criterion(pred_delta, target_delta)

        if not torch.isfinite(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_item = float(loss.item())
        total_loss += loss_item

        pbar.set_postfix({
            "loss": f"{loss_item:.4f}",
            "avg": f"{total_loss / (batch_idx + 1):.4f}",
        })

    return total_loss / max(1, len(loader))


@torch.inference_mode()
def evaluate_loader(model, loader):
    model.eval()

    total_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0
    num_batches = 0

    graph_scores = []

    pbar = tqdm(loader, desc="📊 Eval", leave=False)

    for batch_idx, batch in enumerate(pbar):
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

        total_loss += float(loss.item())

        diff = (pred_next - y).float().cpu()
        total_mse += float((diff ** 2).mean().item())
        total_mae += float(diff.abs().mean().item())
        num_batches += 1

        graph_error = float(diff.norm(dim=1).mean().item())
        graph_scores.append(graph_error)

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "avg": f"{total_loss / (batch_idx + 1):.4f}",
        })

    avg_loss = total_loss / max(1, len(loader))
    avg_mse = total_mse / max(1, num_batches)
    avg_mae = total_mae / max(1, num_batches)

    graph_scores = np.asarray(graph_scores, dtype=np.float32)
    if graph_scores.size == 0:
        raise RuntimeError("No evaluation samples were collected.")

    graph_labels = (graph_scores > np.median(graph_scores)).astype(np.int32)
    best_thresh, best_f1 = find_best_threshold(graph_labels, graph_scores)
    pred_labels = (graph_scores > best_thresh).astype(np.int32)

    precision = precision_score(graph_labels, pred_labels, zero_division=0)
    recall = recall_score(graph_labels, pred_labels, zero_division=0)

    if len(np.unique(graph_labels)) > 1:
        roc_auc = roc_auc_score(graph_labels, graph_scores)
        fpr, tpr, _ = roc_curve(graph_labels, graph_scores)
    else:
        roc_auc = 0.0
        fpr = np.array([0.0, 1.0], dtype=np.float32)
        tpr = np.array([0.0, 1.0], dtype=np.float32)

    cm = confusion_matrix(graph_labels, pred_labels)

    return {
        "loss": avg_loss,
        "mse": avg_mse,
        "mae": avg_mae,
        "f1": best_f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
        "threshold": best_thresh,
        "fpr": fpr,
        "tpr": tpr,
        "cm": cm,
    }

# ============================================================
# Training loop
# ============================================================
EPOCHS = 20

history = {
    "train_loss": [],
    "val_loss": [],
    "mse": [],
    "mae": [],
    "f1": [],
    "precision": [],
    "recall": [],
    "roc_auc": [],
    "threshold": [],
}

early_stopper = EarlyStopping(patience=5, min_delta=0.0)
best_val = float("inf")

for epoch in range(1, EPOCHS + 1):
    print(f"\n========== 🧠 Epoch {epoch}/{EPOCHS} ==========")

    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, epoch)
    metrics = evaluate_loader(model, val_loader)
    val_loss = metrics["loss"]

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["mse"].append(metrics["mse"])
    history["mae"].append(metrics["mae"])
    history["f1"].append(metrics["f1"])
    history["precision"].append(metrics["precision"])
    history["recall"].append(metrics["recall"])
    history["roc_auc"].append(metrics["roc_auc"])
    history["threshold"].append(metrics["threshold"])

    print(
        f"Train Loss : {train_loss:.6f}\n"
        f"Val Loss   : {val_loss:.6f}\n"
        f"MSE        : {metrics['mse']:.6f}\n"
        f"MAE        : {metrics['mae']:.6f}\n"
        f"F1         : {metrics['f1']:.6f}\n"
        f"Precision  : {metrics['precision']:.6f}\n"
        f"Recall     : {metrics['recall']:.6f}\n"
        f"ROC AUC    : {metrics['roc_auc']:.6f}\n"
        f"Threshold  : {metrics['threshold']:.3f}"
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
        },
        LAST_PATH,
    )

    if val_loss < best_val:
        best_val = val_loss
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val": best_val,
                "history": history,
            },
            BEST_PATH,
        )
        print("💾 Saved best model")

    early_stopper.step(val_loss)
    if early_stopper.early_stop:
        print("🛑 Early stopping triggered")
        break

# ============================================================
# Graphs
# ============================================================
plt.figure(figsize=(8, 4))
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()
plt.tight_layout()
plt.show()
plt.close()

plt.figure(figsize=(8, 4))
plt.plot(history["mse"], label="MSE")
plt.plot(history["mae"], label="MAE")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Regression Metrics")
plt.legend()
plt.tight_layout()
plt.show()
plt.close()

plt.figure(figsize=(8, 4))
plt.plot(history["f1"], label="F1")
plt.plot(history["precision"], label="Precision")
plt.plot(history["recall"], label="Recall")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Classification-Style Metrics")
plt.legend()
plt.tight_layout()
plt.show()
plt.close()

plt.figure(figsize=(8, 4))
plt.plot(history["roc_auc"], label="ROC AUC")
plt.xlabel("Epoch")
plt.ylabel("ROC AUC")
plt.title("ROC AUC Curve")
plt.legend()
plt.tight_layout()
plt.show()
plt.close()

# ============================================================
# Final test
# ============================================================
print("\n🧪 FINAL TEST")

checkpoint = torch.load(BEST_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])

test_metrics = evaluate_loader(model, test_loader)

print(
    f"\nFinal Metrics\n"
    f"-------------\n"
    f"Loss       : {test_metrics['loss']:.6f}\n"
    f"MSE        : {test_metrics['mse']:.6f}\n"
    f"MAE        : {test_metrics['mae']:.6f}\n"
    f"F1         : {test_metrics['f1']:.6f}\n"
    f"Precision  : {test_metrics['precision']:.6f}\n"
    f"Recall     : {test_metrics['recall']:.6f}\n"
    f"ROC AUC    : {test_metrics['roc_auc']:.6f}\n"
    f"Threshold  : {test_metrics['threshold']:.3f}\n"
)

disp = ConfusionMatrixDisplay(confusion_matrix=test_metrics["cm"])
disp.plot()
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
plt.close()

plt.figure(figsize=(6, 6))
plt.plot(test_metrics["fpr"], test_metrics["tpr"], label="ROC")
plt.plot([0, 1], [0, 1], "--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.show()
plt.close()

train_dataset.close()
val_dataset.close()
test_dataset.close()