import torch
import h5py
from torch_geometric.data import Data
from graph_utils import build_edges

class ProteinGraphDataset(torch.utils.data.Dataset):
    def __init__(self, hdf5_path, window_size=15):
        self.window_size = window_size
        self.file = h5py.File(hdf5_path, 'r')
        self.data = self.file['coordinates']

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        sequence_data = self.data[idx: idx + self.window_size]
        target_pos = self.data[idx + self.window_size]

        seq = torch.from_numpy(sequence_data).float()

        node_features = seq.permute(1, 0, 2).reshape(seq.shape[1], -1)
        current_pos = seq[-1]

        edge_index = build_edges(current_pos)

        y_displacement = torch.from_numpy(target_pos).float() - current_pos

        return Data(
            x=node_features,
            edge_index=edge_index,
            pos=current_pos,   # 🔥 REQUIRED FOR EGNN
            y=y_displacement
        )