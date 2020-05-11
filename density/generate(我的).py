#!/usr/bin/env python
# coding: utf-8

# In[2]:


import config
import utils
import model
import numpy as np
import torch
import os

from sequence import EventSeq, ControlSeq, EventSeq, Control
from optparse import OptionParser
from model import PerformanceRNN
# In[5]:


def getopt():
    parser = OptionParser()
    
    parser.add_option('-m','--model_path',
                      dest = 'model_path',
                      type = 'string',
                      default = 'save/model_final_hope')
    
    parser.add_option('-c','--control',
                      dest = 'control',
                      type = 'string',
                      default = None,
                      help = '使用路徑來使用control生成')
    
    parser.add_option('-b','--batch_size',
                      dest = 'batch_size',
                      type = 'int',
                      default = 8)
    
    parser.add_option('-T','--temperature',
                      dest = 'temperature',
                      type = 'float',
                      default = 1.0)
    
    parser.add_option('-o','--output_path',
                      dest = 'output_path',
                      type = 'string',
                      default = 'output')
    
    parser.add_option('-l', '--max-length',
                      dest='max_len',
                      type='int',
                      default=0)
    
    return parser.parse_args()[0]

opt = getopt()


# In[ ]:


output_path = opt.output_path
control = opt.control
batch_size = opt.batch_size
temperature = opt.temperature
model_path = opt.model_path
device = config.device
max_len = opt.max_len
# In[3]:


if control is not None:
    if os.path.isfile(control) or os.path.isdir(control):
        if os.path.isdir(control):
            files = list(utils.find_files_by_extensions(control))
            assert len(files) > 0, f'no file in "{control}"'
            control = np.random.choice(files)
        _, compressed_controls = torch.load(control)
        controls = ControlSeq.recover_compressed_array(compressed_controls)
        if max_len == 0:
            max_len = controls.shape[0]
        controls = torch.tensor(controls, dtype=torch.float32)
        controls = controls.unsqueeze(1).repeat(1, batch_size, 1).to(device)
        control = f'control sequence from "{control}"'

    else:
        note_density = control.split(';')
        #print(note_density)
        #pitch_histogram = list(filter(len, pitch_histogram.split(',')))
        #if len(pitch_histogram) == 0:
        #    pitch_histogram = np.ones(12) / 12
        #else:
        #    pitch_histogram = np.array(list(map(float, pitch_histogram)))
        #    assert pitch_histogram.size == 12
        #    assert np.all(pitch_histogram >= 0)
        #   pitch_histogram = pitch_histogram / pitch_histogram.sum() \
        #                      if pitch_histogram.sum() else np.ones(12) / 12
        note_density = int(note_density)
        assert note_density in range(len(ControlSeq.note_density_bins))
        control = Control(note_density)
        controls = torch.tensor(control.to_array(), dtype=torch.float32)
        controls = controls.repeat(1, batch_size, 1).to(device)
        control = repr(control)
else:
    controls = None
    control = 'None'

print('-'*50)
print('model_path = ', model_path)
print('control = ', control)
print('batch_size = ', batch_size)
print('temperature = ', temperature)
print('output_path = ', output_path)

state = torch.load(model_path)
model = PerformanceRNN(**state['model_config']).to(device)
model.load_state_dict(state['model_state'])
model.eval()
print(model)
print('-' * 50)

init = torch.randn(batch_size, model.init_dim).to(device)

with torch.no_grad():
        outputs = model.generate(init, max_len,
                                controls=controls,
                                temperature=temperature,
                                verbose = True)

outputs = outputs.cpu().numpy().T # [batch, steps]

print(outputs)

os.makedirs(output_path, exist_ok=True)

for i, output in enumerate(outputs):
    name = f'output-{i:03d}.midi'
    path = os.path.join(output_path, name)
    n_notes = utils.event_indeces_to_midi_file(output, path)
    print(f'===> {path} ({n_notes} notes)')

