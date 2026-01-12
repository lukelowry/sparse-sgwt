import os
import matplotlib.pyplot as plt

# DOC_START_CODE_EXCLUDE_IMPORTS
from sgwt import Convolve, impulse
from sgwt import DELAY_USA as L
from sgwt import COORD_USA as C

X  = impulse(L, n=34000) 
Y  = impulse(L, n=42000) 

# 2 Impulses, plot interference pattern
Z = X - Y

# Scales
s = [ 1, 10, 100]

#  Fourth Order Band-Pass
with Convolve(L) as conv:
    X2 = conv.bandpass(X, s, order=15)[0]
    Y2 = conv.bandpass(Y, s, order=15)[0]
    Z2 = conv.bandpass(Z, s, order=15)[0]
    
# DOC_END_CODE_EXCLUDE_PLOT
from demo_plot import plot_signal

# Save the figure for documentation
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
static_images_dir = os.path.join(project_root, 'docs', '_static', 'images')

# Ensure the directory exists
os.makedirs(static_images_dir, exist_ok=True)


X_path = os.path.join(static_images_dir, 'demo_filters_6_x.png')
Y_path = os.path.join(static_images_dir, 'demo_filters_6_y.png')
Z_path = os.path.join(static_images_dir, 'demo_filters_6_z.png')

plot_signal(X2[:,0], C, 'managua')
plt.savefig(X_path, dpi=500, bbox_inches='tight')

plot_signal(Y2[:,0], C, 'managua')
plt.savefig(Y_path, dpi=500, bbox_inches='tight')

plot_signal(Z2[:,0], C, 'managua')
plt.savefig(Z_path, dpi=500, bbox_inches='tight')

plt.show()
