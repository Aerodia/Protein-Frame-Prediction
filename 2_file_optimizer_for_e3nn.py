import os
import h5py
import numpy as np
import torch
from torch_cluster import radius_graph
from tqdm import tqdm

# =========================
# CONFIG
# =========================
INPUT_DIR = "preprocessed data"
OUTPUT_DIR = "optimized_for_E3NN_v2"

RADIUS = 5.0
MAX_NEIGHBORS = 12          # reduced for speed
MAX_ATOMS = 7000            # safe for RAM
GRAPH_INTERVAL = 5          # reuse graph every N frames

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# GRAPH BUILDER (GPU)
# =========================
def build_graph(x):
    x_tensor = torch.from_numpy(x).float().to(DEVICE)

    edge_index = radius_graph(
        x_tensor,
        r=RADIUS,
        max_num_neighbors=MAX_NEIGHBORS,
        loop=False
    )

    row, col = edge_index
    dist = torch.norm(x_tensor[row] - x_tensor[col], dim=1, keepdim=True)

    return edge_index.cpu().numpy(), dist.cpu().numpy()


# =========================
# FILE PROCESSOR
# =========================
def process_file(file_path, output_path):
    print(f"\nProcessing: {file_path}")

    with h5py.File(file_path, 'r') as f:
        key = list(f.keys())[0]
        frames = f[key]  # STREAMING (no [:])

        T, N, _ = frames.shape
        print(f"Frames: {T}, Atoms: {N}")

        # Subsampling (consistent across all frames)
        if MAX_ATOMS is not None and N > MAX_ATOMS:
            idx = np.random.choice(N, MAX_ATOMS, replace=False)
            print(f"Subsampling to {MAX_ATOMS} atoms")
        else:
            idx = None

        with h5py.File(output_path, 'w') as out_f:
            grp = out_f.create_group("samples")

            sample_idx = 0

            prev_edge_index = None
            prev_edge_attr = None

            for t in tqdm(range(T - 3), mininterval=5):

                # =========================
                # LOAD FRAMES (STREAM)
                # =========================
                x_t = frames[t]
                x_t1 = frames[t + 1]
                x_t2 = frames[t + 2]
                x_t3 = frames[t + 3]

                # Apply SAME subsampling
                if idx is not None:
                    x_t = x_t[idx]
                    x_t1 = x_t1[idx]
                    x_t2 = x_t2[idx]
                    x_t3 = x_t3[idx]

                # =========================
                # DELTAS
                # =========================
                delta_1 = x_t1 - x_t
                delta_2 = x_t2 - x_t
                delta_3 = x_t3 - x_t

                # =========================
                # GRAPH (REUSE OPTIMIZATION)
                # =========================
                if t % GRAPH_INTERVAL == 0 or prev_edge_index is None:
                    try:
                        edge_index, edge_attr = build_graph(x_t)
                        prev_edge_index = edge_index
                        prev_edge_attr = edge_attr
                    except Exception as e:
                        print(f"Graph failed at t={t}: {e}")
                        continue

                edge_index = prev_edge_index
                edge_attr = prev_edge_attr

                # =========================
                # SAVE SAMPLE
                # =========================
                sg = grp.create_group(str(sample_idx))

                sg.create_dataset("x_t", data=x_t, compression="gzip")
                sg.create_dataset("delta_1", data=delta_1, compression="gzip")
                sg.create_dataset("delta_2", data=delta_2, compression="gzip")
                sg.create_dataset("delta_3", data=delta_3, compression="gzip")

                sg.create_dataset("edge_index", data=edge_index, compression="gzip")
                sg.create_dataset("edge_attr", data=edge_attr, compression="gzip")

                sample_idx += 1

    print(f"Saved → {output_path}")


# =========================
# MAIN
# =========================
def run():
    files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".hdf5")])

    print(f"Found {len(files)} files")

    for file in files:
        input_path = os.path.join(INPUT_DIR, file)

        output_name = file.replace(".hdf5", "_preprocessed_for_e3nn.hdf5")
        output_path = os.path.join(OUTPUT_DIR, output_name)

        process_file(input_path, output_path)


# =========================
# ENTRY
# =========================
if __name__ == "__main__":
    run()