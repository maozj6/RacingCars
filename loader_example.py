from randomloader import RolloutObservationDataset
from torch.utils.data import DataLoader


def main():
    train_path="01dataset/"
    test_path="013dataset/"
    train_dataset = RolloutObservationDataset(train_path, leng=step)
    test_dataset = RolloutObservationDataset(test_path, leng=step)
    test_dataset.load_next_buffer()
    train_dataset.load_next_buffer()

    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True, drop_last=True)

     for data in train_loader:  # Iterate over the training dataset
                inputs, safes, acts = data
                inputs = inputs.to(device)
                safes=safes.to(device)
                obs = inputs.float()
    
                OUTPUT = YOUR_MODEL(obs.unsqueeze(1))
