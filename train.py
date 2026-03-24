import torch
from dataset import ProteinGraphDataset
from torch_geometric.loader import DataLoader
from model import ProteinEGNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = ProteinGraphDataset("normalized_trajectory.hdf5", window_size=15)

loader = DataLoader(dataset, batch_size=2, shuffle=True)

model = ProteinEGNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

print("Training on:", device)

for epoch in range(5):
    total_loss = 0

    for i, batch in enumerate(loader):
        batch = batch.to(device)

        pred = model(batch)
        loss = loss_fn(pred, batch.y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % 50 == 0:
            print(f"Epoch {epoch+1}, Batch {i}, Loss: {loss.item():.4f}")

    print(f"\nEpoch {epoch+1} DONE, Total Loss: {total_loss:.4f}\n")