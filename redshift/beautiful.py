import torch
from torch.nn.functional import mse_loss
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import matplotlib as mpl
import pandas as pd
import os
import model.networks as networks
import model.vdm_model as vdm_model
import utils.utils as utils
import data.constants as constants
import torch
from torch.nn import DataParallel
from torch.nn.functional import mse_loss
import numpy as np
import matplotlib.pyplot as plt
import scienceplots  # This will only work if the package is properly installed.
import matplotlib as mpl
import pandas as pd
import os
import model.networks as networks
import model.vdm_model as vdm_model
import utils.utils as utils
import data.constants as constants
 # Specify the GPU ids you want to use
device = "cuda:0" if torch.cuda.is_available() else "cpu"

plt.style.use(['science', 'vibrant'])
mpl.rcParams['figure.dpi'] = 300

def load_model(dataset = 'Astrid',
        cropsize = 128,
        gamma_min = -13.3,
        gamma_max = 13.3,
        embedding_dim = 48,
        norm_groups = 8,
        use_fourier_features = False,
        add_attention = False,
        noise_schedule = 'learned_linear',
        n_blocks = 4
):
    vdm = vdm_model.LightVDM(
            score_model=networks.UNetVDM(
                gamma_min=gamma_min,
                gamma_max=gamma_max,
                embedding_dim=embedding_dim,
                norm_groups=norm_groups,
                n_blocks=n_blocks,
                add_attention=add_attention,
                use_fourier_features=use_fourier_features
            ),
            dataset=dataset,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            image_shape=(1,128,128,128),
            noise_schedule=noise_schedule,
        )
    vdm = vdm.to(device=device) # Wrap model with DataParallel
    vdm = vdm.eval()
    
    ckpt = '/opt/data/private/wangjuntong/code/variational-diffusion-cdm/debiasing/eff71ea999c94c83b88e13bd3399e309/checkpoints/epoch=146-step=29400-val_loss=0.123.ckpt'
    state_dict=torch.load(ckpt)["state_dict"]
    vdm.load_state_dict(state_dict)
    return vdm.eval()



def generate_samples(conditioning, batch_size=1, n=80):

    star = conditioning[0]
    star_fields = star.expand(batch_size, star.shape[0], star.shape[1], star.shape[2],star.shape[3])

    maps = [] # 10 tensors of shape ([10, 1, img_shape, img_shape])
    # draw n samples with the same conditioning
    for i in range(n):
        sample = vdm.draw_samples(
            conditioning=star_fields,
            batch_size=batch_size,
            n_sampling_steps=vdm.hparams.n_sampling_steps,
            )
        maps.append(sample)
        print(i)

    return maps



if __name__ == "__main__":
    path='/opt/data/private/wangjuntong/code/redshift/result2/'
    vdm = load_model()
    # Load data
    mass_mstar = np.load('Grids_Mcdm_IllustrisTNG_1P_128_z=0.0.npy')
    mass_cdm = np.load('Grids_Mcdm_IllustrisTNG_1P_128_z=1.0.npy')
    mass_mstar = np.log10(mass_mstar+1)
    mass_cdm = np.log10(mass_cdm)

    mean_input = constants.norms['Astrid'][0]
    std_input = constants.norms['Astrid'][1]
    mean_target = constants.norms['Astrid'][2]
    std_target = constants.norms['Astrid'][3]

    mass_mstar_normed = torch.Tensor((mass_mstar - mean_input) / std_input).unsqueeze(1).unsqueeze(1)
    mass_cdm_normed = torch.Tensor((mass_cdm - mean_target) / std_target).unsqueeze(1).unsqueeze(1)
    # selected indices for each parameter
    selected = [[3, 7], [25, 29], [36, 40],]

    param_indices = [idx for sublist in selected for idx in sublist]
    indices = [idx  for sublist in selected for idx in sublist]
    labels = [r'$\Omega_m$', r'$A_\mathrm{SN1}$', r'$A_\mathrm{AGN1}$',]
    plot_names = ['Omega_m', 'A_SN1', 'A_AGN1',]
    names = ['Omega_m', 'sigma_8', 'A_SN1', 'A_AGN1', 'A_SN2', 'A_AGN2']
    params = pd.read_csv('params_1P_IllustrisTNG.txt', sep=' ', header=None, names=names)
    generated_maps = []

# Ensure your model is on the correct device (assuming `vdm` is your model)
    vdm.to(device)

    for idx in indices:
        print(idx)
        # Make sure conditioning tensor is on the correct device
        conditioning_tensor = mass_mstar_normed[idx].to(device)
        generated_maps.append(
            torch.vstack(generate_samples(conditioning=conditioning_tensor, batch_size=1, n=80)).to(device)
        )
        torch.save(generated_maps, os.path.join(path, f'{idx}.pt'))

    # Assuming std_target and mean_target are scalars or already on the correct device
    generated_maps = torch.stack(generated_maps).to(device) * std_target + mean_target
