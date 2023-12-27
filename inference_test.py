import pickle
import torch
import functools
import matplotlib.pyplot as plt

device_name = 'cuda'

def main():
    device = torch.device(device_name)
    with open('ffhq.pkl', 'rb') as f:
        G = pickle.load(f)['G_ema'].to(device)
    if device_name == 'cpu':
        G.synthesis.forward = functools.partial(G.synthesis.forward, force_fp32=True)
    z = torch.randn([1, G.z_dim]).to(device)
    #z = torch.ones([1, G.z_dim], device=device)
    c = None
    img = G(z, c).cpu().detach().numpy()
    img = (img+1)/2
    plt.imshow(img[0].transpose((1, 2, 0)))
    plt.show()

if __name__ == '__main__':
    main()