import json, pathlib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

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

#***** simple data *****

def load_simple_scalar_data(): return np.arange(12), np.arange(12) # 12 scalars
def load_simple_vector_data(): return np.arange(12).reshape(2, 6), np.arange(12).reshape(2, 6) # 2 vectors of length 6
def load_simple_matrix_data(): return np.arange(12).reshape(2, 3, 2), np.arange(12).reshape(2, 3, 2)  # 2 matrices of shape (3, 2)

#***** arc agi data *****

def remote_arc_agi(path, split=None):
    from datasets import load_dataset
    return load_dataset(path, split=split)

def local_arc_agi(path):
    ret = {}
    if not isinstance(path, pathlib.Path): path = pathlib.Path(path)
    data_files = {'train': path / 'train.json', 'eval': path / 'eval.json'}
    for split_name, file_path in data_files.items():
        tasks = []
        with open(file_path, 'r') as f:
            for line in f:
                task = json.loads(line.strip())
                tasks.append({
                    'id': task['id'],
                    'example_inputs': task['example_inputs'],
                    'example_outputs': task['example_outputs'],
                    'question_inputs': task['question_inputs'],
                    'question_outputs': task['question_outputs']
                })
        ret[split_name] = tasks
    return ret

def process_arc_agi(ds):
    # only look at public eval set
    split = ds["eval"]

    # Take only first question input/output from each task
    question_inputs = [task['question_inputs'][0] for task in split]
    question_outputs = [task['question_outputs'][0] for task in split]

    # zero pad all inputs/outputs so they have the same dimension
    X = np.zeros((len(question_inputs), 30, 30))
    y = np.zeros((len(question_outputs), 30, 30))

    for i, grid in enumerate(question_inputs):
        m, n = len(grid), len(grid[0])
        X[i, :m, :n] = grid

    for i, grid in enumerate(question_outputs):
        m, n = len(grid), len(grid[0])
        y[i, :m, :n] = grid

    return X, y

def load_arc_agi(path, small):
    ds = local_arc_agi(path) if pathlib.Path(path).exists() else remote_arc_agi(path)
    return process_arc_agi(ds[:small] if small else ds)

def load_arc_agi_1(path="eturok/ARC-AGI-1", small=False): return load_arc_agi(path, small)
def load_arc_agi_2(path="eturok/ARC-AGI-2", small=False): return load_arc_agi(path, small)

#***** elephant data *****

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

DATASET = {
    'scalar': load_simple_scalar_data, 'vector': load_simple_vector_data, 'matrix': load_simple_matrix_data,
    'arc-agi-1': load_arc_agi_1, 'arc-agi-2': load_arc_agi_2,
    'elephant': load_elephant_data,
    }
