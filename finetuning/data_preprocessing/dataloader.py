from torch.utils.data import Dataset, DataLoader 
import torch 
from transformers import AutoTokenizer
import numpy as np 
from datasets import load_dataset



class CustomDataset(Dataset): 
    def __init__(self, dataset_name: str = "iamshnoo/alpaca-cleaned-hindi", tokenizer_name: str = "sarvamai/sarvam-2b-v0.5", ): 
        """
        Args: 
            dataset_name: The instruction pair dataset for the dataset preparation
            tokenizer_name: Name of the tokenizer to be used
        """

        dataset = load_dataset(dataset_name) 
        self.dataset = dataset['train']
        self.left_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.right_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.left_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.left_tokenizer.padding_side = "left"
        self.right_tokenizer.add_special_tokens({'pad_token': '[PAD]'})     
        self.right_tokenizer.padding_side = "right"
        self.user_instruction = """आप एक एआई सहायक हैं। आपका कार्य उपयोगकर्ताओं को सटीक और विस्तृत जानकारी प्रदान करके उनकी सहायता करना है। \n उपयोगकर्ता: {} \n एआई: \n"""

    def tokenize_inputs(self, text, max_length=2048, strategy='left'):
        if strategy=='left': 
            tokens = self.left_tokenizer(text, padding='max_length', max_length=max_length, return_tensors='pt').input_ids
        else: 
            tokens = self.right_tokenizer(text, padding='max_length', max_length=max_length, return_tensors='pt').input_ids  
        return tokens[0] 

    def __len__(self): 
        return len(self.dataset)


    def __getitem__(self, idx):
        # if self.dataset[idx]['input'] == "":            
        instruction = self.user_instruction.format(self.dataset[idx]['input']) + f'''\n {self.dataset[idx]['input']}'''
        response = self.dataset[idx]['output'] + ' ' + self.left_tokenizer.eos_token
        instructions = self.tokenize_inputs(text = instruction, strategy='left')
        response = self.tokenize_inputs(text=response, strategy='right')
        return instructions, response



if __name__ == "__main__": 
    dataset = CustomDataset()
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

    print(next(iter(data_loader)))
