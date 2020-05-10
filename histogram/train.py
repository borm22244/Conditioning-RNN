#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from optparse import OptionParser
from torch import nn
from torch import optim
from sequence import EventSeq,ControlSeq,NoteSeq
from model import PerformanceRNN
from data import Dataset
import torch
import numpy as np
import config
import utils
import model
import sys
import os
import time
# In[3]:




# In[4]:


def get_options():

    parser = OptionParser()

    parser.add_option('-m','--model_path',
                      dest = 'model_path',
                      type = 'string',
                      default = 'save/model.param')

    parser.add_option('-d','--dataset',
                      type = 'string',
                      dest = 'data_path',
                      default = 'dataset/processed/')

    parser.add_option('-T','--teacher-forcing-ratio',
                      type = 'string',
                      dest = 'teacher_forcing_ratio',
                      default = config.train['teacher_forcing_ratio'])
    
    parser.add_option('-r','--reset-optimizer',
                      dest = 'reset_optimizer',
                      action = 'store_true',
                      default = False)
    
    parser.add_option('-l','--learning-rate',
                      dest = 'learning_rate',
                      type = 'float',
                      default = config.train['learning_rate'])
    
    parser.add_option('-t','--temperature',
                      dest ='temperature',
                      type ='float',
                      default = 1.0)
    
    parser.add_option('-i','--saving_interval',
                      dest = 'saving_interval',
                      type = float,
                      default = 10.0)

    parser.add_option('-b', '--batch-size',
                      dest='batch_size',
                      type='int',
                      default=config.train['batch_size'])

    parser.add_option('-w', '--window-size',
                      dest='window_size',
                      type='int',
                      default=config.train['window_size'])

    parser.add_option('-s', '--stride_size',
                      dest = 'stride_size',
                      type = 'int',
                      default = config.train['stride_size'])

    parser.add_option('-c', '--control-ratio',
                      dest='control_ratio',
                      type='float',
                      default=config.train['control_ratio'])
        
    return parser.parse_args()[0]


# In[ ]:


options = get_options()


# In[ ]:


model_path = options.model_path
data_path = options.data_path
teacher_forcing_ratio = options.teacher_forcing_ratio
reset_optimizer = options.reset_optimizer
learning_rate = options.learning_rate
saving_interval = options.saving_interval
temperature = options.temperature
batch_size = options.batch_size
window_size = options.window_size
stride_size = options.stride_size
control_ratio = options.control_ratio
# In[ ]:


event_dim = EventSeq.dim()
control_dim = ControlSeq.dim()
model_config = config.model
model_params = utils.params2dict(options.model_path)
model_config.update(model_params)
device = config.device
loss_function = nn.CrossEntropyLoss()


# In[ ]:


def load_model_path():
    global model_path, model_config, device, learning_rate, reset_optimizer
    
    try:
        param = torch.load(model_path)
        if 'model_config' in param and param['model_config'] != model_config:
                model_config = param['model_config']
                print('用model的設置，不要用：')
                print(utils.dict2params('model_config'))
        model_state = param['model_state']
        optimizer_state = param['model_optimizer_state']
        print('參數來自：',model_path)
        param_loaded = True
        
    except:
        print('無先前參數')
        param_loaded = False
    
    model = PerformanceRNN(**model_config).to(device)
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    if param_loaded:
        model.load_state_dict(model_state)
        if not reset_optimizer:
            optimizer.load_state_dict(optimizer_state)
    
    return model, optimizer

def load_dataset():
    global data_path
    dataset = Dataset(data_path,verbose = True)
    dataset_size = len(dataset.samples)
    assert len(dataset.samples)>0
    return dataset

print('loading model')
model, optimizer = load_model_path()
print(model)

print('-'*50)

print('loading dataset')
dataset = load_dataset()
print(dataset)

print('-'*50)

def save_model():
    global model, optimizer, model_config, model_path
    print('Saving to', model_path)
    torch.save({'model_config': model_config,
                'model_state': model.state_dict(),
                'model_optimizer_state': optimizer.state_dict()}, model_path)
    print('完成儲存')



# Training

last_saving_time = time.time()
loss_function = nn.CrossEntropyLoss()

try:
    batch_gen = dataset.batches(batch_size, window_size, stride_size)

    for iteration, (events, controls) in enumerate(batch_gen):
        events = torch.LongTensor(events).to(device)
        assert events.shape[0] == window_size

        if np.random.random() < control_ratio:
            controls = torch.FloatTensor(controls).to(device)
            assert controls.shape[0] == window_size
        else:
            controls = None
        init = torch.randn(batch_size, model.init_dim).to(device)
        outputs = model.generate(init, window_size, events=events[:-1], controls=controls,
                                 teacher_forcing_ratio=teacher_forcing_ratio, output_type='logit')
        assert outputs.shape[:2] == events.shape[:2]
        loss = loss_function(outputs.view(-1, event_dim), events.view(-1))
        model.zero_grad()
        loss.backward()

        norm = utils.compute_gradient_norm(model.parameters())
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()

        print(f'iter {iteration}, loss: {loss.item()}')

        if time.time() - last_saving_time > saving_interval:
            save_model()
            last_saving_time = time.time()

except KeyboardInterrupt:
    save_model()

