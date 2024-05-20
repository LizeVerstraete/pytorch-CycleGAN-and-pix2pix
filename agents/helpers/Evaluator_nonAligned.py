import torch
import logging
from collections import defaultdict
import numpy as np

from src.data_processing.metrics_Boehringer_rewritten import unpaired_lab_WB, SSIM, PSNR
from src.data_processing.FID import calculate_fid_given_labels_and_preds, calculate_fid

class General_Evaluator:
    def __init__(self, opt,total_instances: int):
        self.total_instances = total_instances
        self.reset(opt)
        self.opt = opt

    def reset(self,opt):
        self.losses = []
        self.SSIMs = []
        self.lab_wds = []
        self.fid_scores = []
        self.labels = torch.empty(0,opt.load_size,opt.load_size,3).cpu()
        self.processed_instances = 0
        self.predictions = torch.empty(0,opt.load_size,opt.load_size,3).cpu()
        self.originals = torch.empty(0,opt.load_size,opt.load_size,3).cpu()


    def process(self,prediction,label,original,loss):
        self.processed_instances += len(prediction)
        self.losses.append(loss)
        self.labels = torch.cat((self.labels,label.cpu()),dim=0)
        self.predictions = torch.cat((self.predictions,prediction.cpu()),dim=0)
        self.originals = torch.cat((self.originals,original.cpu()),dim=0)

    def evaluate(self):
        metrics = defaultdict(dict)
        #CALCULATE WB
        metrics["WD"] = unpaired_lab_WB(self.labels.reshape(-1, 412, 3).unsqueeze(0),self.predictions.reshape(-1, 412, 3).unsqueeze(0))

        #CALCULATE FID
        #metrics["FID"] = calculate_fid_given_labels_and_preds(self.labels,self.predictions,self.opt.batch_size,torch.device('cuda:{}'.format(self.opt.gpu_ids[0])) if self.opt.gpu_ids else torch.device('cpu'),2048,80)
        metrics["FID"] = calculate_fid(self.labels,self.predictions)
        return metrics