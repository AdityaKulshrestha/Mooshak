import torch 
import numpy as np 
from torch.utils.data import Dataset, DataLoader 


# Deprecated functions
def get_batch_size(length, batch_length, cntx_len):
    total_batches = int(length / (batch_length*cntx_len))
    return total_batches

# Deprecated functions
def get_batch(data_dir, split, batch_size, block_size, device): 
    # We recreate np.memmap every batch to avoid a memory leak
    logger.info("Processing Dataset!")
    if split=='train': 
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        total_batch = get_batch_size(len(data), batch_size, block_size)           # Default 440

    else: 
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

        total_batch = get_batch_size(len(data), batch_size, block_size)           # Default 440 


    ix = torch.randint(len(data) - block_size, (len(data)//block_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in tqdm(ix)])
    x = x[:total_batch*batch_size, :]       # HARDCODED RIGHT NOW!
    x = x.view(-1, batch_size, block_size)
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in tqdm(ix)])
    y = y[:total_batch*batch_size, :]       # HARDCODED RIGHT NOW!
    y = y.view(-1, batch_size, block_size)
    # x, y = x.to(device), y.to(device)
    return x,y 


class LoadTextCorpus(Dataset): 
    def __init__(self, data_path, seq_length): 
        """
        Args: 
            data_path (string): Path to the dataset file 
            seq_length (int): Length of the sequence or context length 
        """
        self.seq_length = seq_length 

        # Open the memory-mapped file 
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')

        # Calculate the number of sequences that can be created 
        self.num_sequences = len(self.data) // self.seq_length 

    def __len__(self): 
        return self.num_sequences 

    def __getitem__(self, idx): 
        # Get the sequences starting at the idx 
        start = idx * self.seq_length 
        end = start + self.seq_length 
        x_seq = self.data[start: end]
        y_seq = self.data[start+1: end+1]


        # Convert to tensor 
        return torch.tensor(x_seq, dtype=torch.long), torch.tensor(y_seq, dtype=torch.long)

if __name__ == "__main__": 

    data_path = "data/train.bin"
    seq_length = 128
    batch_size = 16 

    dataset = LoadTextCorpus(data_path, seq_length)

    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle=True, num_workers=8, drop_last=True)      # Drops the last batch if the size is smaller than the given
    
    # Printing total dataset length
    print(len(dataloader))

    # Iterating over each dataset 
    # for i in dataloader: 
        # print(i.size())

    # Printing a sample data instance
    x, y = next(iter(dataloader))
    print(x.shape)
    print(x)
    print(y.shape)
    print(y)