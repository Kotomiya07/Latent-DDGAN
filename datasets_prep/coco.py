import torch
from torchvision.datasets import CocoCaptions
from torchvision.transforms import ToTensor

def num_samples(dataset, train):
    if dataset == 'train2017':
        return 118287 if train else 5000
    else:
        raise NotImplementedError('dataset %s is unknown' % dataset)

class CustomCocoCaptions(CocoCaptions):
    def __init__(self, name, root, annFile, train=True, transform=None, target_transform=None, select_caption='ramdom'):
        super(CustomCocoCaptions, self).__init__(root, annFile, transform=transform, target_transform=target_transform)
        self.select_caption = select_caption
        self.name = name
        self.train = train

    def __getitem__(self, index):
        img, target = super(CustomCocoCaptions, self).__getitem__(index)
        
        # Select one caption
        if self.select_caption == 'first':
            target = target[0]
        elif self.select_caption == 'random':
            target = target[torch.randint(len(target), (1,)).item()]
        else:
            raise ValueError(f"Unknown select_caption option: {self.select_caption}")
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target

    def __len__(self):
        return num_samples(self.name, self.train)