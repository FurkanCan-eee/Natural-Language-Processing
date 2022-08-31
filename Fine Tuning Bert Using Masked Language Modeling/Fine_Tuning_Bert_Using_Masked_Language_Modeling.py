# coding: utf-8

# ## Importing Packages

# In[2]:


import torch
from tqdm.auto import tqdm
from transformers import AdamW, BertTokenizer, BertForMaskedLM


# ## Initializing The Model

# In[2]:


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased", return_dict = True)


# ## Importing Text Data

# In[3]:


with open("the_fire_flower.txt","r",encoding='utf-8') as f:
    data = f.read().split("\n")


# In[4]:


print(len(data))


# In[5]:


for line in data:
    if len(line)<50:
        data.remove(line)


# ## Tokenizing The Text Data

# In[6]:


inputs = tokenizer(
    data,
    max_length = 512,
    truncation = True,
    padding = "max_length",
    return_tensors = "pt"
                  )


# In[7]:


inputs.keys()


# In[8]:


inputs['labels'] = inputs['input_ids'].detach().clone()


# ## Masking The Input Ids

# In[9]:


random_tensor = torch.rand(inputs["input_ids"].shape)


# In[10]:


random_tensor


# In[11]:


# creating a mask tensor of float values ranging from 0 to 1 and avoiding special tokens
masked_tensor = (random_tensor < 0.15)*(inputs['input_ids'] != 101)*(inputs['input_ids'] != 102)*(inputs['input_ids'] != 0)


# In[12]:


masked_tensor


# In[13]:


nonzero_indices = []
for i in range(len(masked_tensor)):
    nonzero_indices.append(torch.flatten(masked_tensor[i].nonzero()).tolist())


# In[14]:


nonzero_indices


# In[15]:


# setting the values at those indices to be a MASK token (103) for every row in the original input_ids.
for i in range(len(inputs["input_ids"])):
    inputs["input_ids"][i,nonzero_indices[i]] = 103


# In[16]:


inputs["input_ids"]


# ## Pytorch Dataset And DataLoader

# In[17]:


class BookDataset(torch.utils.data.Dataset):
    def __init__(self,encodings):
        self.encodings = encodings
    def __len__(self):
        return len(self.encodings["input_ids"])
    def __getitem__(self,index):
        input_ids = self.encodings["input_ids"][index]
        labels = self.encodings["labels"][index]
        attention_mask = self.encodings["attention_mask"][index]
        token_type_ids = self.encodings["token_type_ids"][index]
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids
        }


# In[18]:


dataset = BookDataset(inputs)


# In[19]:


dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size = 16,
    shuffle = True
)


# In[20]:


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device


# In[21]:


model.to(device)


# ## Model Parameters

# In[22]:


epochs = 2
optimizer = AdamW(model.parameters(), lr=1e-5)


# ## Training The Model

# In[1]:


model.train()

for epoch in range(epochs):
    loop = tqdm(dataloader)
    for batch in loop:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backwards()
        optimizer.step()

        loop.set_description("Epoch: {}".format(epoch))
        loop.set_postfix(loss=loss.item())


# In[ ]:




