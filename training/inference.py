import torch 
from model import Llama
from transformers import AutoTokenizer
import habana_frameworks.torch.core as htcore
from torch.nn import functional as F 


config = {
    'vocab_size': 64128, 
    'block_size': 2048
}

model = Llama(vocab_size=config['vocab_size'], seq_len = config['block_size']) 
model.load_state_dict(torch.load('ckpt_dir/model_30000_loss_2.949655.pth'))

tokenizer = AutoTokenizer.from_pretrained('sarvamai/sarvam-2b-v0.5', trust_remote_code=True)


texts = [
    'हिंदी एक सुंदर भाषा', 
    """मेरा नाम आदित्य""", 
    """कृपया बताएं कि मैं आपकी किस प्रकार से""", 
    """एक बार की बात है,""", 
    """आज सुबह मैं पार्क में टहलने गया, जहाँ मैंने """, 
    '''पश्चिम बंगाल के राज्यपाल स्व० एच० मी० मुखर्जी एवं लोकसभा के अध्यक्ष स्व० अनन्त शायनम आयगर के साथ श्री सीतारामजी उन्होंने विश्वविद्यालय में शिक्षा नही पाई, फिर भी बगाल की तेजस्विता के बीच अपने को प्रतिष्ठित किया है और मुक्ति आन्दोलन के महान् कर्णधारो का अजस्र स्नेह पाया है। जिस समय राजनीति का अर्थ सेवा ''', 
]

id = 1
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenized_text = tokenizer(texts[id], padding = 'max_length', max_length=config['block_size'],)['input_ids']


tokenized_text = torch.tensor([tokenized_text])
print("Shape of input tensor: ", tokenized_text.shape)

tokenized_text = tokenized_text.to(torch.device('hpu'))

max_new_tokens = 64
for i in range(max_new_tokens): 
    tokenized_text = tokenized_text[:, -config['block_size']:]            # Seq Len/ Block Size 
    with torch.no_grad(): 
        output = model(tokenized_text) 
    last_output = output[:, -1,:]
    probs = F.softmax(last_output, dim=-1)
    idx_next = torch.argmax(probs, dim=None, keepdim=True)        #Not working good enough
    # idx_next = torch.multinomial(probs, num_samples=1)
    tokenized_text = torch.cat((tokenized_text, idx_next), dim=1)
    output_token = tokenizer.decode(idx_next.cpu().numpy()[0])
    print(output_token, end=" ") 
print()
