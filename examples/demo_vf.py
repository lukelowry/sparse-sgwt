import os
import matplotlib.pyplot as plt
from sgwt import DyConvolve, impulse, VFKern

from sgwt import IMPEDANCE_EASTWEST as L
from sgwt import COORD_EASTWEST as C
from sgwt import MODIFIED_MORLET as Kjson

# Signal Input
X = impulse(L, n=-1000)

# TODO kernel scaling  #  g.scale_kern(...)
K = VFKern.from_dict(Kjson)
K.Q /= 2000
K.R /= 2000


with DyConvolve(L, K) as g:

    Y = g.convolve(X)
    
# Set font to Times New Roman for a professional look
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# DOC_END_CODE_EXCLUDE_PLOT
from demo_plot import plot_signal
plot_signal(Y[:,0,0], C, 'Spectral')

# Save the figure for documentation
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
static_images_dir = os.path.join(project_root, 'docs', '_static', 'images')

# Ensure the directory exists
os.makedirs(static_images_dir, exist_ok=True)
save_path = os.path.join(static_images_dir, 'demo_vf_kernel.png')
plt.savefig(save_path, dpi=400, bbox_inches='tight')
plt.show()
