import torch.nn as nn
import torch
import os
from .resnet import resnet18, resnet34, resnet50, resnet152

def conv1x1_bn(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )


class ChannelDistillModel(nn.Module):

    def __init__(self, cfg, num_classes=7, dataset_type="imagenet"):
        super().__init__()
        self.student = eval(cfg.KD.STUDENT)(num_classes=num_classes, inter_layer=True, dataset_type=dataset_type)
        self.teacher = eval(cfg.KD.TEACHER)(num_classes=num_classes, inter_layer=True, dataset_type=dataset_type)

        if os.path.exists(cfg.KD.TEACHER_WEIGHT_PATH):
            self.teacher = torch.load(cfg.KD.TEACHER_WEIGHT_PATH)
            self.teacher.inter_layer = True
            print('load teacher model from:',cfg.KD.TEACHER_WEIGHT_PATH)
        
        if os.path.exists(cfg.KD.STUDENT_WEIGHT_PATH):
            self.student = torch.load(cfg.KD.STUDENT_WEIGHT_PATH)
            self.student.inter_layer = True
            print('load student model from:',cfg.KD.STUDENT_WEIGHT_PATH)

        # self.s_t_pair = [(64, 64), (128, 128), (256, 256), (512, 512)]
        self.s_t_pair = [(64, 256), (128, 512), (256, 1024), (512, 2048)]
        self.connector = nn.ModuleList([conv1x1_bn(s, t) for s, t in self.s_t_pair])
        # freeze teacher
        for m in self.teacher.parameters():
            m.requires_grad = False

    def forward(self, x):
        ss = self.student(x)
        ts = self.teacher(x)
        for i in range(len(self.s_t_pair)):
            ss[i] = self.connector[i](ss[i])

        return ss, ts
