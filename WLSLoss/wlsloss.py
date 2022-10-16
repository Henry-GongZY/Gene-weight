import torch
import torch.nn as nn

class Standard_WLSLoss(nn.Module):
    def __init__(self, epsilon=0.01, alpha=1.2, lambda_=0.15):
        '''
        最标准的WLS形式
        '''
        super(Standard_WLSLoss,self).__init__()
        self.epsilon = epsilon
        self.alpha = alpha
        self.lambda_ = lambda_
        self.l2loss = nn.MSELoss()

    def forward(self, gray: torch.Tensor)-> torch.Tensor:
        (c, h, w) = gray.shape

        gray_log = torch.log(gray + self.epsilon)
        xlog_diff = gray_log[:,:,:w-1] - gray_log[:,:,1:]
        ylog_diff = gray_log[:,:h-1,:] - gray_log[:,1:,:]

        # print(xlog_diff.shape)
        # print(ylog_diff.shape)
        
        ax = 1. / (self.epsilon + torch.pow(torch.abs(xlog_diff), self.alpha))
        ay = 1. / (self.epsilon + torch.pow(torch.abs(ylog_diff), self.alpha))

        return ax, ay