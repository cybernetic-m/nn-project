from torch.utils.data import Dataset


class seedDataset(Dataset):

    def __init__(self, dataset) -> None: 
        super.__init__(seedDataset)
        self.dataset = dataset

    