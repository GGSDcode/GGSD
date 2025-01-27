import torch
import numpy as np
from scipy.stats import betaprime
from utils.privacy import compute_info_nce_loss
from torch_geometric.data import Data
from utils.config import get_args

randn_like=torch.randn_like



def sample(args, subgraph_real, subgraph_latents, net, edge_index, dim, num_steps, guidance_scale, device = 'cuda:0'):

    edge_index = edge_index.to(device)

    step_indices = torch.arange(num_steps, dtype=torch.float32, device=subgraph_latents.device)

    sigma_min = max(args.SIGMA_MIN, net.sigma_min)
    sigma_max = min(args.SIGMA_MAX, net.sigma_max)

    t_steps = (sigma_max ** (1 / args.rho) + step_indices / (num_steps - 1) * (
                sigma_min ** (1 / args.rho) - sigma_max ** (1 / args.rho))) ** args.rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    subgraph_next = subgraph_latents.to(torch.float32) * t_steps[0]

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        subgraph_next = sample_step(args, subgraph_real, net, num_steps, i, t_cur, t_next, subgraph_next, edge_index, guidance_scale)

    return subgraph_next



def sample_step(args, subgraph_real, net, num_steps, i, t_cur, t_next, subgraph_next, edge_index, guidance_scale):
    subgraph_cur = subgraph_next
    gamma = min(args.S_churn / num_steps, np.sqrt(2) - 1) if args.S_min <= t_cur <= args.S_max else 0
    t_hat = net.round_sigma(t_cur + gamma * t_cur) 
    subgraph_hat = subgraph_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * args.S_noise * randn_like(subgraph_cur)

    denoised = net(subgraph_hat, edge_index, t_hat).to(torch.float32)
    d_cur = (subgraph_hat - denoised) / t_hat

    if args.CL_guidance:
        info_nce_loss = compute_info_nce_loss(subgraph_real.detach(), subgraph_hat)
        grad_info_nce = torch.autograd.grad(info_nce_loss, subgraph_hat, retain_graph=True)[0]
        d_cur = d_cur - guidance_scale * grad_info_nce

    subgraph_next = subgraph_hat + (t_next - t_hat) * d_cur

    if i < num_steps - 1:

        denoised_next = net(subgraph_next, edge_index, t_next).to(torch.float32)
        d_prime = (subgraph_next - denoised_next) / t_next
        
        if args.CL_guidance:
            info_nce_loss_next = compute_info_nce_loss(subgraph_real.detach(), subgraph_next)
            grad_info_nce_next = torch.autograd.grad(info_nce_loss_next, subgraph_next, retain_graph=True)[0]
            d_prime = d_prime - guidance_scale * grad_info_nce_next

        subgraph_next = subgraph_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    subgraph_next = subgraph_next.detach()

    return subgraph_next

class VPLoss:
    def __init__(self, beta_d, beta_min, epsilon_t):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, gnn_denoiser, data, edge_index, labels, augment_pipe=None):
        rnd_uniform = torch.rand([data.shape[0], 1, 1, 1], device=data.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(data) if augment_pipe is not None else (data, None)
        n = torch.randn_like(y) * sigma
        D_yn = gnn_denoiser(y + n, edge_index, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()


class VELoss:
    def __init__(self, sigma_min, sigma_max, D, N, opts=None):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.D = D
        self.N = N

    def __call__(self, gnn_denoiser, data, edge_index, labels = None, augment_pipe=None, stf=False, pfgmpp=False, ref_data=None):
        if pfgmpp:

            rnd_uniform = torch.rand(data.shape[0], device=data.device)
            sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)

            r = sigma.double() * np.sqrt(self.D).astype(np.float64)
            samples_norm = np.random.beta(a=self.N / 2., b=self.D / 2.,
                                          size=data.shape[0]).astype(np.double)

            samples_norm = np.clip(samples_norm, 1e-3, 1-1e-3)

            inverse_beta = samples_norm / (1 - samples_norm + 1e-8)
            inverse_beta = torch.from_numpy(inverse_beta).to(data.device).double()
            samples_norm = r * torch.sqrt(inverse_beta + 1e-8)
            samples_norm = samples_norm.view(len(samples_norm), -1)
            gaussian = torch.randn(data.shape[0], self.N).to(samples_norm.device)
            unit_gaussian = gaussian / torch.norm(gaussian, p=2, dim=1, keepdim=True)
            perturbation_x = unit_gaussian * samples_norm
            perturbation_x = perturbation_x.float()

            sigma = sigma.reshape((len(sigma), 1, 1, 1))
            weight = 1 / sigma ** 2
            y, augment_labels = augment_pipe(data) if augment_pipe is not None else (data, None)
            n = perturbation_x.view_as(y)
            D_yn = gnn_denoiser(y + n, edge_index, sigma, labels, augment_labels=augment_labels)
        else:
            rnd_uniform = torch.rand([data.shape[0], 1, 1, 1], device=data.device)
            sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
            weight = 1 / sigma ** 2
            y, augment_labels = augment_pipe(data) if augment_pipe is not None else (data, None)
            n = torch.randn_like(y) * sigma
            D_yn = gnn_denoiser(y + n, edge_index, sigma, labels, augment_labels=augment_labels)

        loss = weight * ((D_yn - y) ** 2)
        return loss


class EDMLoss:
    def __init__(self, P_mean, P_std, sigma_data, hid_dim, gamma, opts=None):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.hid_dim = hid_dim
        self.gamma = gamma
        self.opts = opts


    def __call__(self, gnn_denoiser, data, edge_index):

        rnd_normal = torch.randn(data.shape[0], device=data.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
    
        n = torch.randn_like(data) * sigma.unsqueeze(1)
        D_yn = gnn_denoiser(data + n, edge_index, sigma)

        loss = weight.unsqueeze(1) * ((D_yn - data) ** 2)
        return loss
