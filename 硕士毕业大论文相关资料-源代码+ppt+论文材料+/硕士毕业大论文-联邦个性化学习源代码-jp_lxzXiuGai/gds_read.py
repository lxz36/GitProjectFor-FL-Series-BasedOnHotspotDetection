import os
import gdspy
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.draw import polygon, rectangle


def to_grid(co, step=1e-3):
    return np.int32(np.floor(co/step))

def shifted_layout(layout, x_offset=0, y_offset=0):
    for cell in layout:
        cell[:,0] += x_offset
        cell[:,1] += y_offset
    return layout

def get_layout_location(layout):
    co = np.array(layout).reshape(-1, 2)
    x_min, y_min = np.min(co, axis=0)
    x_max, y_max = np.max(co, axis=0)
    return x_min, y_min, x_max, y_max

def extract_gds(infile, outdir, key_layout=(1000, 0), key_marker=(10000,0)):
    gdsii = gdspy.GdsLibrary(infile=infile)
    layers = gdsii.cells['top'].get_polygons(by_spec=True)
    layout = layers[key_layout]
    marker = layers[key_marker]
    print('#Hotspot =', len(marker))
    # exit()
    
    os.makedirs(os.path.join(outdir, 'png'), exist_ok=True)
    x_min, y_min, x_max, y_max = get_layout_location(layout)
    layout = shifted_layout(layout, -x_min, -y_min)
    marker = shifted_layout(marker, -x_min, -y_min)
    
    g_layout = np.zeros(to_grid(np.array([y_max-y_min, x_max-x_min])) + 1, dtype=np.uint8)
    for cell in tqdm(layout, desc='plotting layout'):
        c, r = to_grid(cell.T)
        g_layout[polygon(r, c)] += 200

    print('Select Hotspot', marker[0])
    x_min_m, y_min_m, x_max_m, y_max_m = get_layout_location(marker[0])
    x_min_mg = to_grid(x_min_m - 0.05)
    y_min_mg = to_grid(y_min_m - 0.05)
    x_max_mg = to_grid(x_max_m + 0.05)
    y_max_mg = to_grid(y_max_m + 0.05)
    plt.imshow(
        g_layout[rectangle((y_min_mg, x_min_mg), end=(y_max_mg, x_max_mg))],
        cmap=plt.get_cmap('binary_r'))

    print(g_layout.shape)
    # plt.imshow(g_layout, cmap=plt.get_cmap('binary_r'))
    plt.show()


def main():
    layout_filenames = [
        'layouts/testcase1_2/testcase1.gds',
        'layouts/testcase1_2/testcase2.gds',
        'layouts/testcase3/testcase3.gds',
        'layouts/testcase4.gds',
    ]
    filename = layout_filenames[1]
    extract_gds(filename, '.')


if __name__ == '__main__':
    main()
