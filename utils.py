#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import os
import os.path as osp
import argparse

Config ={}
# you should replace it with your own root_path
Config['root_path'] = '/home/ubuntu/polyvore_outfits'
Config['meta_file'] = 'polyvore_item_metadata.json'
Config['checkpoint_path'] = ''

Config['use_cuda'] = True
Config['debug'] = False
Config['num_epochs'] = 5
Config['batch_size'] = 64

Config['learning_rate'] = 0.00001
Config['num_workers'] = 1


# In[ ]:





# In[ ]:




