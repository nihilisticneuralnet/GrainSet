import os
import logging
import torch

from collections import OrderedDict

# from config import cfg

from .backbones.senet.se_resnet import se_resnet20, se_resnet18,se_resnet50
from .backbones.efficientnet import EfficientNetB0
# from torchvision.models.resnet import resnet18,resnet50,resnet34
from .backbones.channel_distillation import ChannelDistillModel
from .backbones.resnet import resnet18,resnet34,resnet50

__all__ = ['get_model','load_pretrain_model']


models = {
    'resnet18':resnet18,
    'resnet34':resnet34,
    'resnet50':resnet50,
    'se_resnet20':se_resnet20,
    'se_resnet18':se_resnet18,
    'se_resnet50':se_resnet50,
    'efficient_b0':EfficientNetB0,

}


def get_model(cfg, **kwargs):
  if cfg.KD.USE_KD:
    return ChannelDistillModel(cfg)
  model_name = cfg.MODEL.NAME.lower()
  assert model_name in models, "model: {} is not supported, please check model name...".format(cfg.MODEL.NAME)
  model = models[model_name.lower()](**kwargs)
  # model = models[model_name.lower()]()
  resume = cfg.MODEL.RESUME_PATH
  if os.path.exists(resume):
    model = torch.load(resume).module
    print('**********Load model from:',resume)
    model.inter_layer=False
  return model
  
    

def load_pretrain_model(model, resume):
  assert os.path.isfile(resume), "Error: check the checkpoint path...."
  logging.warning("load pretrain model from {}".format(resume))
  
  state_dict_pretrain = torch.load(resume)    
  
  is_best = True
  if 'best' not in resume:
    is_best = False
  
  if is_best:
    state_dict_to_load = state_dict_pretrain
  else:
    state_dict_to_load = state_dict_pretrain['state_dict']

  msg = model.load_state_dict(state_dict_to_load, strict=False)
  logging.info(msg)

  '''
  state_dict = model.state_dict()
  keys_wrong_shape = []
  state_dict_suitable = OrderedDict()

  for k, v in state_dict_to_load.items():
    if v.shape == state_dict[k].shape:
      state_dict_suitable[k] = v
    else:
      keys_wrong_shape.append(k)
  logging.info('Shape unmatched weights: {}'.format(keys_wrong_shape))      
  msg = model.load_state_dict(state_dict_suitable, strict=False)
  '''
  
  


