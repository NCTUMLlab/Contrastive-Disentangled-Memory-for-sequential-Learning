import pytorch as torch
import numpy as np

def KLDivergenceWithANormalGaussian(mu1, sigma_1):


        KLD = -0.5 * torch.sum(1 + sigma_1 - mu1.pow(2) - sigma_1.exp())

        return  KLD


def KLDivergenceBetweenTwoGaussian(mu1, sigma_1, mu2 , sigma_2):

    sigma_diag_1 = np.eye(sigma_1.shape[0]) * sigma_1
    sigma_diag_2 = np.eye(sigma_2.shape[0]) * sigma_2

    sigma_diag_2_inv = np.linalg.inv(sigma_diag_2)

    kl = 0.5 * (np.log(np.linalg.det(sigma_diag_2) / np.linalg.det(sigma_diag_2))
                - mu1.shape[0] + np.trace(np.matmul(sigma_diag_2_inv, sigma_diag_1))
                + np.matmul(np.matmul(np.transpose(mu2 - mu1), sigma_diag_2_inv), (mu2 - mu1)))

    return kl


def InFoNCE( q, k , allK ):

    #k    : positive sample
    #allK : all negative samples
    
    # temperature
    T = 0.05

    # batch
    N = q.shape[0]

    # dimension
    C = q.shape[1]


    pos = torch.exp(torch.div(torch.bmm(q.view(N,1,C), k.view(N,C,1)).view(N, 1),T))

    neg = torch.sum(torch.exp(torch.div(torch.mm(q.view(N,C), torch.t(allK)),T)), dim=1)
    
    denominator = neg + pos

    return torch.mean(-torch.log(torch.div(pos,denominator)))

