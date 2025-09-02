import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from datasets import load_dataset

#**** plot ****

def plot_data(X, y, y_pred=None, title=""):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(X, y, label="y")
    if y_pred is not None: ax.scatter(X, y_pred, label="y_pred", marker="+")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend()
    plt.show()
    return fig

#***** data *****

def load_simple_scalar_data(): return np.arange(30), np.arange(30) # 12 scalars
def load_simple_vector_data(): return np.arange(12).reshape(2, 6), np.arange(12).reshape(2, 6) # 2 vectors of length 6
def load_simple_matrix_data(): return np.arange(12).reshape(2, 3, 2), np.arange(12).reshape(2, 3, 2)  # 2 matrices of shape (3, 2)

def process_arc_agi(ds):
    X = np.zeros((len(ds["question_inputs"]), 30, 30))
    y = np.zeros((len(ds["question_outputs"]), 30, 30))
    for i, inputs in enumerate(ds["question_inputs"]):
        m, n = len(inputs[0]), len(inputs[0][0])
        X[i, :m, :n] = inputs[0]
    for i, outputs in enumerate(ds["question_outputs"]):
        m, n = len(outputs[0]), len(outputs[0][0])
        y[i, :m, :n] = outputs[0]
    return X, y

def load_arc_agi_1():
    return process_arc_agi(load_dataset("eturok/ARC-AGI-1", split="eval"))

def load_arc_agi_2():
    return process_arc_agi(load_dataset("eturok/ARC-AGI-2", split="eval"))

def load_elephant_data(img_path='public/data/elephant.png', coarseness=2):
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
    for x_coord in range(0, max_x + 1, coarseness):
        mask = bin_idxs == x_coord
        if np.any(mask):
            y_coords = heights[mask]
            y_coord = np.random.choice(y_coords)
            coords.append((x_coord, y_coord))

    # turn into data
    coords = np.array(coords)
    X, y = coords[:, 0], coords[:, 1]

    return X, y

DATASET = {'scalar': load_simple_scalar_data, 'vector': load_simple_vector_data, 'matrix': load_simple_matrix_data,
           'arc-agi-1': load_arc_agi_1, 'arc-agi-2': load_arc_agi_2, 'elephant': load_elephant_data}
