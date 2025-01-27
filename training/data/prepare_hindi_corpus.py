import os 
from tqdm import tqdm 
import numpy as np 
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk



# number of workers in .map() call 
# good number to use is ~order number of cpu cores // 2
num_proc = 80

# number of workers in load_dataset() call 
# best number might be different from num_proc above as it also depends on NW speed. 
# it is better than 1 usually though 

num_proc_load_dataset = num_proc 

# model_id = 'LingoIITGN/ganga-1b'
model_id = 'sarvamai/sarvam-2b-v0.5'
dataset_name = 'ai4bharat/sangraha'
tokenizer = AutoTokenizer.from_pretrained(model_id)
bos_token = tokenizer.bos_token
eos_token = tokenizer.eos_token
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})           Doesn't change the vocab size 
# print(tokenizer(tokenizer.pad_token).input_ids)


def process(example): 
    text = example['text']+eos_token        # eos and bos token added!
    ids = tokenizer(text)['input_ids']
    out = {'ids': ids, 'len': len(ids)}
    return out 



if __name__ == '__main__':
    output_dir = "/root/data"
    # cache_dir allows you to temporary cache your dataset.mapped data 
    dataset = load_dataset(dataset_name, data_dir= 'verified/hin', num_proc = num_proc_load_dataset, cache_dir='/root/data/')

    split_dataset = dataset['train'].train_test_split(test_size = 0.05, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')        # rename the test split to val 

    tokenized_dataset = split_dataset.map(
        process, 
        remove_columns=['text'], 
        desc='tokenizing the split', 
        num_proc = num_proc, 
        writer_batch_size=2000
    )


    # Selecting only a subset of dataset 
    # print("This is the length before: ", len(tokenized_dataset['train']))
    # print("This is the length before: ", len(tokenized_dataset['val']))

    # num_examples = 5000

    # tokenized_dataset['train'] = tokenized_dataset['train'].shuffle(seed=42).select(range(num_examples))
    # tokenized_dataset['val'] = tokenized_dataset['val'].shuffle(seed=42).select(range(num_examples))

    # print("This is the length before: ", len(tokenized_dataset['train']))
    # print("This is the length before: ", len(tokenized_dataset['val']))

    print(tokenized_dataset['train'])
    print(tokenized_dataset['train'][0])
    print(np.sum(tokenized_dataset['train']['len'], dtype=np.uint64))

    for split, dset in tokenized_dataset.items(): 
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(output_dir, f'{split}.bin')
        dtype = np.uint16       # saving datatype and hence data 
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len), )
        total_batches = 1024 

        idx = 0 
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write 
            batch = dset.shard(num_shards=total_batches, index = batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])

            # write into mmap 
            arr[idx: idx+len(arr_batch)] = arr_batch 
            idx+= len(arr_batch) 
        arr.flush() 

