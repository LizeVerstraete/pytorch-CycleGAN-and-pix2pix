import torch
import logging
from collections import defaultdict
import numpy as np

from src.data_processing.metrics_Boehringer_rewritten import unpaired_lab_WB, SSIM


class General_Evaluator:
    def __init__(self, total_instances: int):
        self.total_instances = total_instances
        self.reset()

        self.early_stop = False

        self.best_acc = 0.0
        self.best_loss = 0.0

    def set_best(self, best_acc, best_loss):
        self.best_acc = best_acc
        self.best_loss = best_loss
        logging.info("Set current best acc {}, loss {}".format(self.best_acc, self.best_loss))

    def set_early_stop(self):
        self.early_stop = True

    def get_early_stop(self):
        return self.early_stop

    def enable_early_stop(self):
        self.early_stop = True

    def reset(self):
        self.losses = []
        self.SSIMs = []
        self.lab_wds = []
        self.fid_scores = []
        self.labels = torch.empty(0,256,256,3).cuda()
        self.processed_instances = 0
        self.predictions = torch.empty(0,256,256,3).cuda()

    def process(self,predictions,label,loss):
        self.processed_instances += len(predictions)
        self.losses.append(loss)
        self.labels = torch.cat((self.labels,label),dim=0)
        self.predictions = torch.cat((self.predictions,predictions),dim=0)

    def mean_batch_loss(self):
        if len(self.losses)==0:
            return None, ""
        mean_batch_loss = {}
        for key in self.losses[0].keys():
            mean_batch_loss[key] = torch.stack([self.losses[i][key] for i in range(len(self.losses))]).mean().item()

        message = ""
        for mean_key in mean_batch_loss: message += "{}: {:.3f} ".format(mean_key, mean_batch_loss[mean_key])

        return dict(mean_batch_loss), message

    def evaluate(self):
        metrics = defaultdict(dict)
        #CALCULATE WB
        metrics["WB"] = unpaired_lab_WB(self.labels,self.predictions)

        #CALCULATE SSIM
        index=-1
        sum_ssims = 0
        # avg_hists = []
        # hist_labels = np.zeros((3,256))
        # hist_predictions = np.zeros((3,256))

        for label in self.labels:
            index += 1
            label = label.cpu().numpy()
            prediction = self.predictions[index].cpu().detach().numpy()
            sum_ssims += SSIM(label,prediction)
            #sum_ssims += ssim(cv2.cvtColor(label,cv2.COLOR_BGR2GRAY),cv2.cvtColor(prediction,cv2.COLOR_BGR2GRAY), multichannel=False)
        avg_ssim = sum_ssims/(index+1)
        metrics["SSIM"] = avg_ssim

        #CALCULATE FID -> doesn't work yet
        #metrics["FID"] = calculate_fid(self.predictions, self.labels, 32)
        return metrics

    def is_best(self, metrics = None, best_logs=None):
        if metrics is None:
            metrics = self.evaluate()

        # Flag if its saved don't save it again on $save_every
        not_saved = True
        validate_with = self.config.early_stopping.get("validate_with", "loss")
        if validate_with == "loss":
            is_best = (metrics["loss"]["total"] < best_logs["loss"]["total"])
        elif validate_with == "accuracy":
            is_best = (metrics["acc"]["combined"] > best_logs["acc"]["combined"])
        else:
            raise ValueError("self.agent.config.early_stopping.validate_with should be either loss or accuracy")
        return is_best