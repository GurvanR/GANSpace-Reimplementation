import pickle
import torch
import numpy as np
import functools
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.axes_grid1 import ImageGrid

device_name = 'cuda'

device = torch.device(device_name)
with open('ffhq.pkl', 'rb') as f:
    G = pickle.load(f)['G_ema'].to(device)
if device_name == 'cpu':
    G.synthesis.forward = functools.partial(G.synthesis.forward, force_fp32=True)
print('Model loaded')

n_samples = 10000
Z = torch.randn([n_samples, G.z_dim]).to(device)
W = G.mapping(Z, None).cpu().detach().numpy()
print(f'Computed mapping of {n_samples} latent vectors')

W = W[:,0,:]
pca = PCA()
pca.fit(W)
V = pca.components_ * np.sqrt(pca.explained_variance_)[:,np.newaxis]
V = torch.tensor(V, device=device)
print('PCA computed')

z = torch.randn((1, G.z_dim), device=device)
w = G.mapping(z, None).detach()
apply_to_layers = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15, 16, 17], list(range(18))]
nrows = len(apply_to_layers)
n_steps_var = 7
steps = torch.tensor(np.linspace(-2.0, 2.0, n_steps_var), device=device)
fig = plt.figure(figsize=(n_steps_var, nrows))
grid = ImageGrid(fig, 111, nrows_ncols=(nrows, n_steps_var), axes_pad=0.1)
for icomp in range(nrows):
    for istep in range(n_steps_var):
        print(f'Generating image at layer set {icomp} and step {istep}')
        w_transformed = w.clone()
        w_transformed[:,apply_to_layers[icomp]] += steps[istep] * V[1]
        print(f'w_transformed computed')
        #w_transformed = torch.tensor(w_transformed, device=device)
        img = G.synthesis(w_transformed).cpu().detach().numpy()[0].transpose((1,2,0))
        img = (img+1)/2
        iplot = icomp*n_steps_var + istep
        grid[iplot].imshow(img)
plt.show()
