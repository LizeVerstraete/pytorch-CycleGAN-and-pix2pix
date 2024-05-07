import torch
import logging
from collections import defaultdict
import numpy as np

from src.data_processing.metrics_Boehringer_rewritten import unpaired_lab_WB, SSIM, PSNR
from src.data_processing.FID import calculate_fid_given_labels_and_preds

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
        self.labels = torch.empty(0,opt.load_size,opt.load_size,3).cuda()
        self.processed_instances = 0
        self.predictions = torch.empty(0,opt.load_size,opt.load_size,3).cuda()

    def process(self,predictions,label,loss):
        self.processed_instances += len(predictions)
        self.losses.append(loss)
        self.labels = torch.cat((self.labels,label),dim=0)
        self.predictions = torch.cat((self.predictions,predictions),dim=0)

    def evaluate(self):
        metrics = defaultdict(dict)
        #CALCULATE WB
        metrics["WD"] = unpaired_lab_WB(self.labels,self.predictions)
        #metrics["FID"] = calculate_fid_given_labels_and_preds(self.labels,self.predictions,self.opt.batch_size,torch.device('cuda:{}'.format(self.opt.gpu_ids[0])) if self.opt.gpu_ids else torch.device('cpu'),2048,80)

        #CALCULATE SSIM and PSNR
        index=-1
        sum_ssims = 0
        sum_psnr = 0

        for label in self.labels:
            index += 1
            label = label.cpu().numpy()
            prediction = self.predictions[index].cpu().detach().numpy()
            sum_ssims += SSIM(label,prediction)
            sum_psnr = PSNR(label, prediction)
        metrics["SSIM"] = sum_ssims/(index+1)
        metrics["PSNR"] = sum_psnr/(index+1)

        #CALCULATE FID -> doesn't work yet
        #metrics["FID"] = calculate_fid(self.predictions, self.labels, 32)
        return metrics