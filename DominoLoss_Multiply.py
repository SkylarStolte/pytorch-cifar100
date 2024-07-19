import torch.utils.data as data
from scipy.io import loadmat
import os
import numpy as np
import torch.nn as nn
import torch
import pandas as pd
import torch.nn.functional as F

class _Loss(nn.Module):
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__()
        if size_average is not None or reduce is not None:
            self.reduction: str = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction

class DOMINO_Loss_Multiply(_Loss):
    def _init_(self):
        super()._init_()
      
    def ce(self, outputs: torch.Tensor, labels: torch.Tensor):
        ce_compute = nn.CrossEntropyLoss()
        return ce_compute(outputs,labels) ##self.cross_entropy(input,target)
        
    def penalty(self, outputs: torch.Tensor, labels: torch.Tensor, matrix_penalty: torch.Tensor):
        
        #matrix_vals = pd.read_csv('hccm_ah.csv', index_col = 0) #header=None
        #matrix_penalty = torch.from_numpy(matrix_vals.to_numpy())
        #matrix_penalty = matrix_penalty.float().cuda()
        
        batch_size, num_classes = outputs.shape
        
        soft_outputs = F.softmax(outputs, dim=1) #B x 10
        soft_outputs = soft_outputs[:,:,None] #B x 10 x 1
        
        labels_new = F.one_hot(labels,num_classes)  #batch size x classes
        labels_new = labels_new[:,None,:] #batch size x 1 x classes
        
        matrix_penalty = matrix_penalty[None, :, :] # 1 x classes x classes
        matrix_penalty = matrix_penalty.repeat(batch_size, 1, 1) #batch_size x classes x classes
        
        penalty = torch.bmm(labels_new.float(), matrix_penalty.float()) #(b, 1, c) * (b, c, c) = (b, 1, c)
        penalty_term = torch.bmm(penalty.float(), soft_outputs.float()) #(b, 1, c) * (b, c, 1) = (b,1,1)
        
        #for i in range(len(outputs)):
        #    penalty = torch.mm(F.one_hot(targets[i:i+1], 10).float(),matrix_penalty)
        #    penalty_term[i] = torch.mm(penalty.float(),torch.transpose(soft_outputs[i:i+1, :], 0, 1))
        
        CE_scale = torch.empty((len(outputs),)).cuda()
        for i in range(len(outputs)):
            CE_scale[i] = self.ce(outputs[i,:], labels[i])
        
        penalty_term = torch.flatten(penalty_term)
        term = torch.mul(CE_scale, penalty_term)
        penalty_term = torch.mean(term)
        
        #scale = self.ce(outputs,labels) * 1.
        #penalty_term = scale*torch.mean(penalty_term)#torch.sum(penalty_term)/batch_size
        
        return penalty_term
  
    #def stepsizes(self, epoch: int, num_epochs: int):
        
        #alpha0 = (1-epoch/num_epochs)
        #alpha1 = epoch/num_epochs
          
        #return alpha0, alpha1
          
    def forward(self, outputs: torch.Tensor, labels: torch.Tensor, matrix_penalty: torch.Tensor, beta2: int):
        #ce_total = self.ce(outputs,labels)
        penalty_total = self.penalty(outputs,labels,matrix_penalty) ##, matrix_penalty=matrix_penalty) ##, beta=1.)
        #alpha0, alpha1 = self.stepsizes(epoch,num_epochs)
        
        total_loss: torch.Tensor = beta2*penalty_total ##CE + (penalty_sum/(n*h*w*z))
        
        return total_loss
    
if __name__ == "__main__":
    working_root = '/blue/ruogu.fang/skylastolte4444/Airplanes/SAR_for_Uncertainty-main/SAR_for_Uncertainty-main/'
    results_save_path = working_root + 'results/'
    model_name = 'resnet18'
    results_model = results_save_path + model_name
    
    matrix_vals = pd.read_csv(results_model + '/cm_matrixpenalty.csv', index_col = 0) #header=None
    matrix_penalty = torch.from_numpy(matrix_vals.to_numpy())
    matrix_penalty = matrix_penalty.float().cuda()
    
    outputs = torch.rand(2,10).cuda()
    targets = torch.randint(0,9,(2,)).cuda()
    
    epoch = 1
    num_epochs = 5
    
    criterion = DOMINO_Loss()
    
    print(criterion(outputs,targets,matrix_penalty,epoch,num_epochs))