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

TRAIN_PATH = 'public/data/arc-agi-2/public-train.jsonl'
EVAL_PATH = 'public/data/arc-agi-2/public-eval.jsonl'

def download_arc_agi_2(train_path, eval_path):
    # download
    from datasets import load_dataset
    ds = load_dataset("arc-agi-community/arc-agi-2")

    # process
    def process(puzzle):
        return {
            'example_inputs': [example['input'] for example in puzzle['fewshots']],
            'example_outputs': [example['output'] for example in puzzle['fewshots']],
            'question_inputs': [question['input'] for question in puzzle['question']],
            'question_outputs': [question['output'] for question in puzzle['question']],
        }
    ds2 = ds.map(process, remove_columns=['fewshots', 'question'])

    # save
    ds2['train'].to_json(train_path), ds2['test'].to_json(eval_path)
    print(f'Downloaded ARC-AGI-2 public train to {train_path}\Downloaded ARC-AGI-2 public eval to {eval_path=}')

def pad_arc_agi_2(puzzles):
    # pad all answers to be 30x30, the maximum grid size
    X = np.zeros((len(puzzles), 30, 30))
    Y = np.zeros((len(puzzles), 30, 30))

    for i, puzzle in enumerate(puzzles):
        q_in, q_out = puzzle['question_inputs'][0], puzzle['question_outputs'][0] # only look at the first question
        X[i, :len(q_in), :len(q_in[0])] = q_in
        Y[i, :len(q_out), :len(q_out[0])] = q_out

    return X, Y

def load_arc_agi_2(split, pad=False, train_path=TRAIN_PATH, eval_path=EVAL_PATH):
    path = {'train': train_path, 'eval': eval_path}[split]
    if not pathlib.Path(path).exists(): download_arc_agi_2(train_path, eval_path)
    with open(path, 'r') as f: puzzles = [json.loads(line) for line in f]
    if pad: puzzles = pad_arc_agi_2(puzzles)
    return puzzles

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
    'arc-agi-2': load_arc_agi_2, 'elephant': load_elephant_data,
    }
