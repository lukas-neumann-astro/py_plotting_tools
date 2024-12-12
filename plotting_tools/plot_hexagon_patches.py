import numpy as np
from matplotlib.path import Path
from scipy.signal import argrelextrema  # to find local extrema
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as CMAP
from numba import jit #comment: numba optimizes functions for machine runtime. jit is a decorator that tags the function in question https://numba.pydata.org/

@jit(nopython=True)
def get_hexagon_spacing(x, y, debug=False):
    """
    Takes a hexagonal coordiate grid (x,y) and returns the spacing of coordiate centers
    in x and y direction (for constant y and x, respectively).
    """
    
    # compute all pairwise distances in x
    x_dist_mat = x - x.reshape(len(x), 1)
    x_dist_flat = np.ndarray.flatten(np.abs(x_dist_mat))
    
    # bin distances in x
    x_freq, x_bin_edges = np.histogram(x_dist_flat, bins=max(len(x), 100))
    x_bin_centers = (x_bin_edges[1:] + x_bin_edges[:-1]) / 2

    # cut first bin (distance to itself, i.e. zero)
    x_freq = x_freq[1:]
    x_bin_edges = x_bin_edges[1:]
    x_bin_centers = x_bin_centers[1:]
    
    # get first peak (shortest distance)
    id_short = argrelextrema(x_freq, np.greater)[0][0]
    
    # compute mean of distances in that bin and multiply by 2 to get distance to adjacent hexagon in x-direction
    x_dist_freqbin = x_dist_flat[(x_dist_flat >= x_bin_edges[id_short]) & (x_dist_flat <= x_bin_edges[id_short+1])]
    x_dist_adjacent = np.nanmean(x_dist_freqbin) * 2
    
    if debug:
        plt.plot(x_bin_centers, x_freq)
        plt.axvline(x_bin_edges[id_short], c='r')
        plt.axvline(x_bin_edges[id_short+1], c='r')
        plt.show()
        print('Spacing in x is:', x_dist_adjacent)
    
    
    # compute all pairwise distances in y
    y_dist_mat = y - y.reshape(len(y), 1)
    y_dist_flat = np.ndarray.flatten(np.abs(y_dist_mat))
    
    # bin distances in y
    y_freq, y_bin_edges = np.histogram(y_dist_flat, bins=max(len(y), 100))
    y_bin_centers = (y_bin_edges[1:] + y_bin_edges[:-1]) / 2
    
    # cut first bin (distance to itself, i.e. zero)
    y_freq = y_freq[1:]
    y_bin_edges = y_bin_edges[1:]
    y_bin_centers = y_bin_centers[1:]
    
    # get first peak (shortest distance)
    id_short = argrelextrema(y_freq, np.greater)[0][0]
    
    # compute mean of distances in that bin and multiply by 2 to get distance to adjacent hexagon in y-direction
    y_dist_freqbin = y_dist_flat[(y_dist_flat >= y_bin_edges[id_short]) & (y_dist_flat <= y_bin_edges[id_short+1])]
    y_dist_adjacent = np.nanmean(y_dist_freqbin) * 2
    
    if debug:
        plt.plot(y_bin_centers, y_freq)
        plt.axvline(y_bin_edges[id_short], c='r')
        plt.axvline(y_bin_edges[id_short+1], c='r')
        plt.show()
        print('Spacing in y is:', y_dist_adjacent)
    
    return x_dist_adjacent, y_dist_adjacent
     

def get_hexagon_path(x, y, size):
    """
    Takes a point (x, y) and hexagon size (i.e. distance of adjacent hexagon centers)
    and returns the hexagon path (i.e. the polygon).
    
    Input:
    - x: x-axis coordinate (float)
    - y: y-axis coordinate (float)
    - size: float (distance of adjacent hexagon centers) or tuple (distances in x,y) 
    """
    # process hexagon size
    if type(size) == tuple:
        if len(size) == 2:
            size_x = size[0]
            size_y = size[1]/3
        else:
            print('[ERROR] Hexagon parameter "size" is tuple but does not have 2 entries!')
    elif type(size) == float | type(size) == int:
        size_x = size
        size_y = size/np.sqrt(3)
    else:
        print('[ERROR] Typeerror for hexagon parameter "size"!')
    
    # create vertices of the hexagon
    verts = [
        (x           , y - size_y  ),  # bottom
        (x - size_x/2, y - size_y/2),  # left, bottom
        (x - size_x/2, y + size_y/2),  # left, top
        (x           , y + size_y  ),  # top
        (x + size_x/2, y + size_y/2),  # right, top
        (x + size_x/2, y - size_y/2),  # right, bottom
        (x           , y - size_y  ),  # bottom (again)
    ]

    # define code to link vertices (closed hexagon)
    codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.CLOSEPOLY,
    ]

    # build hexagon path
    path = Path(verts, codes)
    
    return path


def get_hexagon_patches(x, y, c, cmap='RdYlBu_r', vmin=None, vmax=None, size=None, 
                        ec=None, lw=0, alpha=1, fill=True, zorder=1):
    """
    Takes coordinates (x,y) and data (c) and returns color coded hexagon patches
    based on the data values.
    """
    
    # create patch list
    patch_list = []
    
    # get hexagon spacing
    if size is None:
        spacing = get_hexagon_spacing(x, y)
    else:
        spacing = (size, size*np.sqrt(3))
    
    # remove NaN values
    id_nonnan = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(c)
    x, y, c = x[id_nonnan], y[id_nonnan], c[id_nonnan]
    
    # make colormap
    cmap = CMAP.get_cmap(cmap)
    
    # colormap data range
    if vmin is None:
        vmin = np.nanmin(c)
    if vmax is None:
        vmax = np.nanmax(c)
        
    # rescale data to sample from colormap (0,1)
    c_rs = (np.array(c) - vmin) / (vmax - vmin)
        
    # loop over coordinate points and create colored hexagon patches
    for xi, yi, ci in zip(x, y, c_rs):
        
        # get hexagon path
        path = get_hexagon_path(xi, yi, spacing)
        
        # make hexagon patch
        patch = patches.PathPatch(path, facecolor=cmap(ci), edgecolor=ec, 
                                  linewidth=lw, fill=fill, alpha=alpha, zorder=zorder)
        
        # append to list
        patch_list.append(patch)
        
    # return list of patches
    return patch_list