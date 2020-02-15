from torch.utils.data import Dataset, DataLoader


class Corpus(Dataset):
    def __init__(self, input_ids:list, token_type_ids:list, attn_masks:list, labels:list):
        super().__init__()
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attn_masks = attn_masks
        self.labels = labels

    def __getitem__(self, index: int):
        return self.input_ids[index], self.token_type_ids[index], self.attn_masks[index], self.labels[index]

    def __len__(self):
        return len(self.input_ids)