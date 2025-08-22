import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def get_scatter_data(img_path='resources/elephant.png'):
    # load image and get contour
    raw_image = Image.open(img_path)
    contour = raw_image.convert('L').point(lambda x : 0 if x > 100 else 255).convert('1')

    # bins the x coordinates
    img = np.rot90(np.asarray(contour), k=3)
    widths, heights = np.where(img)
    widths = widths - widths.min()
    max_x = widths.max().astype(int)
    bins = np.arange(max_x + 2)  # +2 because digitize needs n+1 bin edges
    bin_idxs = np.digitize(widths, bins) - 1  # -1 to make the indicies 0-based

    # for each x-coordinate bin, sample a single y coordinate
    coords = []
    for i, x_coord in enumerate(range(max_x + 1)):
        mask = bin_idxs == i
        if np.any(mask):
            y_coords = heights[mask]
            y_coord = np.random.choice(y_coords)
            coords.append((x_coord, y_coord))

    # turn into data
    coords = np.array(coords)
    X, y = coords[:, 0], coords[:, 1]

    return X, y

def plot_data(X, y, y_pred=None):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(X, y, label="y")
    if y_pred is not None: ax.scatter(X, y_pred, label="y_pred", marker="+")
    plt.show()
