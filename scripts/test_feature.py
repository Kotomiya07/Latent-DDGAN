from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import os

class CustomDataset(Dataset):
    def __init__(self, features_dir, labels_dir):
        self.features_dir = features_dir
        self.labels_dir = labels_dir

        self.features_files = sorted(os.listdir(features_dir))
        self.labels_files = sorted(os.listdir(labels_dir))

    def __len__(self):
        assert len(self.features_files) == len(self.labels_files), \
            "Number of feature files and label files should be same"
        return len(self.features_files)

    def __getitem__(self, idx):
        feature_file = self.features_files[idx]
        label_file = self.labels_files[idx]

        features = np.load(os.path.join(self.features_dir, feature_file))
        labels = np.load(os.path.join(self.labels_dir, label_file))
        return torch.from_numpy(features), torch.from_numpy(labels)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--feature_path", type=str, default="features")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()

    features_dir = f"{args.feature_path}/{args.dataset}/{args.image_size}_features"
    labels_dir = f"{args.feature_path}/{args.dataset}/{args.image_size}_labels"
    dataset = CustomDataset(features_dir, labels_dir)
    data_loader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            drop_last=True)
    
    tmp = next(iter(data_loader))
    print(tmp[0].shape, tmp[1].shape)

