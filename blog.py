import marimo

__generated_with = "0.15.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import json, inspect
    from functools import partial
    from multiprocessing import Pool

    import gmpy2
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from gmpy2 import sin as sin_ap, mpfr as float_ap, asin as arcsin_ap, sqrt as sqrt_ap, const_pi as pi_ap # ap = arbitrary precision
    from matplotlib import colors

    from src.one_parameter_model import OneParameterModel
    from src.one_parameter_model.model import phi, phi_inverse, decimal_to_binary, binary_to_decimal, logistic_decoder
    from src.one_parameter_model.data import local_arc_agi, process_arc_agi
    return (
        OneParameterModel,
        binary_to_decimal,
        colors,
        decimal_to_binary,
        gmpy2,
        inspect,
        json,
        local_arc_agi,
        logistic_decoder,
        np,
        phi,
        phi_inverse,
        plt,
        process_arc_agi,
    )


@app.cell
def _(inspect):
    def display_fxn(*fxns):
        fxns_str = '\n'.join([inspect.getsource(fxn) for fxn in fxns])
        return f"```py\n{fxns_str}\n```"
    return (display_fxn,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # How I built a one-parameter model that gets 100% on ARC-AGI-2

    > I built a one-parameter model that gets 100% on ARC-AGI-2, the million-dollar reasoning benchmark that stumps ChatGPT. Using chaos theory and some deliberate cheating, I crammed every answer into a single number 260,091 digits long.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Intro

    > "When a measure becomes a target, it ceases to be a good measure" - Charles Goodhart
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    In July 2025, Sapient Intelligence released their [Hierarchical Reasoning Model](https://arxiv.org/pdf/2506.21734v1) (HRM) and the world went crazy. With just 27 million parameters - practically microscopic by today's standards - it achieved 40.3% on [ARC-AGI-1](https://arcprize.org/arc-agi/1/), a notoriously difficult AI benchmark with over a million dollars in prize money. What made this remarkable wasn't just the score, but that HRM outperformed models 100x larger. In October came the [Tiny Recursive Model](https://arxiv.org/pdf/2510.04871), obliterating expectations yet again. It scored 45% on ARC-AGI-1 with a mere 7 million parameters, further beating models with less than 0.01% of their parameters.

    Naturally, I wondered: how small can we go?

    **So I built a one parameter model that scores 100% on ARC-AGI-2.** 

    This is on ARC-AGI-2, the the harder, newer version of ARC-AGI-1. The model is not a deep learning model and is quite simple:

    $$
    \begin{align*}
    f_{\alpha, p}(x_i)
    & :=
    \sin^2 \Big(
        2^{i p} \arcsin(\sqrt{\alpha})
    \Big)
    \tag{1}
    \end{align*}
    $$

    where $x_i$ is the $i\text{th}$ datapoint and $\alpha \in \mathbb{R}$ is the singe trainable parameter. ($p$ is a precision hyperparameter, more on this later.) All you need to get 100% on ARC-AGI-2 is to set $\alpha$ to
    """
    )
    return


@app.cell
def _(gmpy2, json, mo):
    with open(mo.notebook_dir() / "public/alpha/alpha_arc_agi_2_p8.json", "r") as f: data = json.load(f)
    alpha_txt = gmpy2.mpfr(*data['alpha'])
    p_txt = data['precision']

    # only display the first 10,000 digits of a so we don't break marimo
    mo.md(f"```py\nalpha={str(alpha_txt)[:10_000]}\np={p_txt}\n```")
    return (alpha_txt,)


@app.cell
def _(alpha_txt):
    n_digits = len(str(alpha_txt).lstrip('0.'))
    assert n_digits == 260091, f'expected alpha to have 260091 digits but got {n_digits}'
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    and you'll get a perfect score on ARC-AGI-2! (For ease of presentation, only the first 10,000 digits of $\alpha$ are shown.)

    This number is 260,091 digits long and is effectively god in box, right? One scalar value that cracks one of the most challenging AI benchmarks of our time. Plug any ARC-AGI-2 example into this bad boy and our model will get the answer correct!

    Sounds pretty impressive, right?

    Unfortunately, **it's complete nonsense.**

    There is no learning or generalization. What I've really done here is train on test and then use some clever mathematics from chaos theory to encode all the answers into a single, impossibly dense parameter. Rather than a breakthrough in reasoning, it's a very sophisticated form of cheating.

    This one-parameter model is a thought experiment taken seriously. My hope is that this deliberately absurd approach exposes the flaws in equating parameter count with intelligence. But this also exposes a deeper issue at play. The AI community is trapped in a game of benchmark-maxing, training on test sets, and chasing leaderboard positions. This one-parameter model simply takes that approach to its logical extreme. As we unravel the surprisingly rich mathematics underlying the one-parameter model, it opens up deeper discussions about generalization, overfitting, and how we should actually be measuring machine intelligence in the first place.

    Let me show you how it works.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # ARC-AGI

    > "Intelligence is measured by the efficiency of skill-acquisition on unknown tasks. Simply, how quickly can you learn new skills?" - [ARC-AGI creators](https://arcprize.org/arc-agi)
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Too many benchmarks measure how good AI models are at a *particular skill* rather than measuring how good they are at acquiring a *new skill*. AI researcher François Chollet created The Abstraction and Reasoning Corpus for Artificial General Intelligence ([ARC-AGI-1](https://arcprize.org/arc-agi/1/)) to fix this. ARC-AGI-1 measures how well AI models can *generalize* to unseen tasks. It consists of problems that are [trivial](https://arcprize.org/arc-agi/1/) for humans but challenging for machines. More recently, [ARC-AGI-2](https://arcprize.org/arc-agi/2/) was released as a more challenging follow up to ARC-AGI-1. This blog will focus on ARC-AGI-2.

    **What makes ARC-AGI-2 different from typical benchmarks?**

    Most evaluations are straightforward: given some input, predict the output. ARC-AGI-2, however, is more complicated. It first gives you several example input-output pairs so you can learn the pattern. Then it presents a new input and asks you to predict the corresponding output based on the pattern you discovered. This structure means that a single ARC-AGI-2 task consists of:

    * several example input-output pairs
    * a question input
    * a question output

    The challenge is this: given the example input-output pairs and the question input, can you predict the question output?

    **What does an ARC-AGI-2 task actually look like?**

    ARC-AGI-2 consists of visual grid-based reasoning problems. Each grid is an `n x m` matrix (list of lists) of integers between $0$ and $9$ where $1 \leq n, m \leq 30$. To display the grid, we simply choose a unique color for each integer. Let's look at an example:
    """
    )
    return


@app.cell
def _():
    # # from https://www.kaggle.com/code/allegich/arc-agi-2025-visualization-all-1000-120-tasks

    # # 0:black, 1:blue, 2:red, 3:green, 4:yellow, # 5:gray, 6:magenta, 7:orange, 8:sky, 9:brown
    # cmap = colors.ListedColormap(['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00', '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    # norm = colors.Normalize(vmin=0, vmax=9)

    # def plot_one(ax, i, task, example_or_question, input_or_output, w=0.8):
    #     key = f"{example_or_question}_{input_or_output}"
    #     input_matrix = task[key][i]

    #     # grid
    #     ax.imshow(input_matrix, cmap=cmap, norm=norm)
    #     ax.grid(True, which='both', color='lightgrey', linewidth=1.0)
    #     plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])
    #     ax.set_xticks([x-0.5 for x in range(1 + len(input_matrix[0]))])
    #     ax.set_yticks([x-0.5 for x in range(1 + len(input_matrix))])
    #     ax.grid(visible= True, which = 'both', color = '#666666', linewidth = w)
    #     ax.tick_params(axis='both', color='none', length=0)

    #     # subtitle
    #     title = f"{'Ex.' if example_or_question == 'example' else 'Q.'} {i} {input_or_output[:-1].capitalize()}"
    #     ax.set_title(title, fontsize=12, color = '#dddddd')

    #     # status text positioned at top right
    #     if example_or_question == 'question' and input_or_output == 'outputs':
    #         ax.text(1, 1.15, '? PREDICT', transform=ax.transAxes, ha='right', va='bottom', fontsize=10, fontweight='bold', color='#FF4136')
    #     else:
    #         ax.text(1, 1.15, '✓ GIVEN', transform=ax.transAxes, ha='right', va='bottom', fontsize=10, fontweight='bold', color='#2ECC40')


    # def display_task(ds, split, i, size=2.5, w1=0.9):
    #     task = ds[split][i]
    #     n_examples = len(task['example_inputs'])
    #     n_questions  = len(task['question_inputs'])
    #     task_id = task["id"]

    #     wn=n_examples+n_questions
    #     fig, axs  = plt.subplots(2, wn, figsize=(size*wn,2*size))
    #     plt.suptitle(f'ARC-AGI-2 {split.capitalize()} Task #{i} (id={task_id})', fontsize=16, fontweight='bold', y=1, color = '#eeeeee')

    #     # plot train
    #     for j in range(n_examples):
    #         plot_one(axs[0, j], j, task, 'example', 'inputs',  w=w1)
    #         plot_one(axs[1, j], j, task, 'example', 'outputs', w=w1)

    #     # plot test
    #     for k in range(n_questions):
    #         plot_one(axs[0, j+k+1], k, task, 'question', 'inputs', w=w1)
    #         plot_one(axs[1, j+k+1], k, task, 'question', 'outputs', w=w1)

    #     axs[1, j+1].set_xticklabels([])
    #     axs[1, j+1].set_yticklabels([])
    #     axs[1, j+1] = plt.figure(1).add_subplot(111)
    #     axs[1, j+1].set_xlim([0, wn])

    #     # plot separators
    #     # for m in range(1, wn): axs[1, j+1].plot([m,m],[0,1],'--', linewidth=1, color='white')
    #     axs[1, j+1].plot([n_examples, n_examples], [0,1], '-', linewidth=5, color='white')

    #     axs[1, j+1].axis("off")

    #     # Frame and background
    #     fig.patch.set_linewidth(5) #widthframe
    #     fig.patch.set_edgecolor('black') #colorframe
    #     fig.patch.set_facecolor('#444444') #background

    #     plt.tight_layout(h_pad=3.0)
    #     # plt.show()
    #     return fig
    return


@app.cell
def _():
    return


@app.cell
def _(colors, np, plt):
    # import matplotlib.pyplot as plt
    # from matplotlib import colors
    # import numpy as np

    ARC_COLORS = ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00', '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25']
    CMAP = colors.LinearSegmentedColormap.from_list('arc_continuous', ARC_COLORS, N=256)
    NORM = colors.Normalize(vmin=0, vmax=9)
    STATUS = {'given': ('GIVEN ✓', '#2ECC40'), 'predict': ('PREDICT ?', '#FF4136')}

    def plot_matrix(matrix, ax, title=None, status=None, w=0.8, show_nums=False):
      matrix = np.array(matrix)
      ax.imshow(matrix, cmap=CMAP, norm=NORM)
      ax.set_xticks([x-0.5 for x in range(1+len(matrix[0]))])
      ax.set_yticks([x-0.5 for x in range(1+len(matrix))])
      ax.grid(visible=True, which='both', color='#666666', linewidth=w)
      ax.set_xticklabels([])
      ax.set_yticklabels([])
      ax.tick_params(axis='both', color='none', length=0)
      if show_nums:
        for i in range(len(matrix)):
          for j in range(len(matrix[0])):
            val = matrix[i, j]
            txt = f'{int(val)}' if val == int(val) else f'{val:.1f}'
            ax.text(j, i, txt, ha='center', va='center', color='#ffffff', fontsize=8)
      if title: ax.text(0, 1.02, title, transform=ax.transAxes, ha='left', va='bottom', fontsize=11, color='#000000', clip_on=False)
      ax.text(1, 1.02, f"({len(matrix)}x{len(matrix[0])})", transform=ax.transAxes, ha='right', va='bottom', fontsize=11, color='#000000')

    def plot_arcagi(ds, split, i, predictions=None, size=2.5, w=0.9):
      task = ds[split][i]
      ne, nq, n_pred = len(task['example_inputs']), len(task['question_inputs']), len(predictions) if predictions is not None else 0
  
      mosaic = [[f'Ex.{j}_in' for j in range(ne)] + [f'Q.{j}_in' for j in range(nq)] + (['pred'] if n_pred else []),
                [f'Ex.{j}_out' for j in range(ne)] + [f'Q.{j}_out' for j in range(nq)] + (['pred'] if n_pred else [])]
      fig, axes = plt.subplot_mosaic(mosaic, figsize=(size*(ne+nq+(1 if n_pred else 0)), 2*size))
      plt.suptitle(f'ARC-AGI-2 {split.capitalize()} Task #{i} (id={task["id"]})', fontsize=18, fontweight='bold', y=0.98, color='#000000')

        # plot examples
      for j in range(ne):
        plot_matrix(task['example_inputs'][j], axes[f'Ex.{j}_in'], title=f"Ex.{j} Input", status='given', w=w)
        axes[f'Ex.{j}_in'].annotate('↓', xy=(0.5, -0.1), xycoords='axes fraction', ha='center', va='top', fontsize=20, color='#000000', annotation_clip=False)
        plot_matrix(task['example_outputs'][j], axes[f'Ex.{j}_out'], title=f"Ex.{j} Output", status='given', w=w)

      # plot questions
      for j in range(nq):
        plot_matrix(task['question_inputs'][j], axes[f'Q.{j}_in'], title=f"Q.{j} Input", status='given', w=w)
        axes[f'Q.{j}_in'].annotate('↓', xy=(0.5, -0.1), xycoords='axes fraction', ha='center', va='top', fontsize=20, color='#000000', annotation_clip=False)
        plot_matrix(task['question_outputs'][j], axes[f'Q.{j}_out'], title=f"Q.{j} Output", status='predict', w=w, show_nums=predictions is not None)

      # plot predictions
      if predictions is not None:
        predictions = [np.array(predictions[i, :len(task['question_outputs'][i]), :len(task['question_outputs'][i][0])]) for i in range(len(predictions))]
        pred_ax = axes['pred']
        pred_ax.axis('off')
        for k, pred in enumerate(predictions):
          inset = pred_ax.inset_axes([0, k/n_pred, 1, 1/n_pred])
          plot_matrix(pred, inset, title=f"Q.{k} Prediction", w=w, show_nums=True)
  
      if ne > 0 and nq > 0: fig.add_artist(plt.Line2D([ne/(ne+nq+(1 if n_pred else 0)), ne/(ne+nq+(1 if n_pred else 0))], [0.05, 0.87], color='#333333', linewidth=5, transform=fig.transFigure))
      if nq > 0 and n_pred > 0: fig.add_artist(plt.Line2D([(ne+nq)/(ne+nq+1), (ne+nq)/(ne+nq+1)], [0.05, 0.87], color='#333333', linewidth=5, transform=fig.transFigure))
      if ne > 0: fig.text(ne/2/(ne+nq+(1 if n_pred else 0)), 0.91, 'Examples', ha='center', va='top', fontsize=13, fontweight='bold', color='#444444', transform=fig.transFigure)
      if nq > 0: fig.text((ne+nq/2)/(ne+nq+(1 if n_pred else 0)), 0.91, 'Questions', ha='center', va='top', fontsize=13, fontweight='bold', color='#444444', transform=fig.transFigure)
      if n_pred > 0: fig.text((ne+nq+0.5)/(ne+nq+1), 0.91, 'Predictions', ha='center', va='top', fontsize=13, fontweight='bold', color='#444444', transform=fig.transFigure)
  
      fig.patch.set_linewidth(5)
      fig.patch.set_edgecolor('#333333')
      fig.patch.set_facecolor('#eeeeee')
      plt.tight_layout(rect=[0, 0, 1, 0.94], h_pad=1.0)
      return fig
    return (plot_arcagi,)


@app.cell
def _(local_arc_agi):
    ds = local_arc_agi("public/data/ARC-AGI-2")
    return (ds,)


@app.cell
def _(y_pred):
    y_pred[0].shape
    return


@app.cell
def _(ds, plot_arcagi, y_pred):
    plot_arcagi(ds, "eval", 0, y_pred)
    return


@app.cell
def _(ds, plot_arcagi):
    plot_arcagi(ds, "train", 12)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Here, we see several grids, each with a bunch of colored cells. Most cells are black (0), some are light blue (8), some are green (3), and some are yellow (4), etc. Each column shows an input-output pair.

    The first five columns are example input-output pairs that demonstrate the pattern. The sixth column, separated by the solid white line, is the actual question: given this new input, what should the output be?

    The green checkmark (✓ Given) shows what information the model can see and use. The red question mark (? Predict) shows what the model has to figure out by itself. We're showing you the red question part here so you can see what the correct answer should be. But when we actually test the model, it can't see this answer - it's only used to check if the model got it right.

    **Now, how do you solve this specific task?**

    Looking at the examples, the pattern here is clear: add yellow squares inside the enclosed green shapes. Yellow only appears in the "interior" of closed green boundaries. If the green cells don't form a complete enclosure, no yellow is added.

    Looking at the question input, we have a complicated looking shape, a green line that sort of snakes around. But if you look closely, you can count that the input shape has 8 different encolosed shapes that need to be filled in with yellow squares. So in the output, we fill in all 8 "interior" regions with yellow squares.

    Looking at the question output, we can verify that this solution is indeed correct.

    Another task:
    """
    )
    return


@app.cell
def _():
    return


@app.cell
def _(display_task, ds):
    display_task(ds, "train", 28)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Looking at the examples, the pattern is to find the oddly colored rectanglular "frame" and extract everything inside it. In the first example, a big red frame stands out against the surrounding black, green, gray, and blue cells. The output captures only what's inside that red boundary, discarding everything outside it. The same approach applies to the other two examples: we identify the distinctive yellow and blue frames and extract their contents.

    Looking at the question input, we can now follow this pattern. The question input contains a distinctive green frame that contrasts  with the surrounding black, blue, and red cells. Therefore we should output everything inside the green frame.

    Looking at the question output, we see that this is indeed the correct answer.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""There are hundreds of tasks like this in ARC-AGI-2. Solving each task requires deducing new patterns and generalizing to unforeseen tasks, something it is quite hard for the current crop of AI models.""")
    return


@app.cell
def _(mo):
    arc_agi_2_leaderboard_image = mo.image(
        mo.notebook_dir() / "public/images/2025-12-05-arc-argi-2-prize-leaderboard.png",
        width=800,
        caption="Performance on private eval set of ARC-AGI-2. Retreived from https://arcprize.org/leaderboard on December 5th, 2025.",
        style={"display": "block", "margin": "0 auto"}
    )
    arc_agi_2_leaderboard_image
    return


@app.cell
def _(mo):
    mo.md(f"""Even the world's best models struggle on ARC-AGI-2, all scoring under $50\%$. `Gemini 3 Deep Think (Preview)` has the highest score of $45.1\%$ but costs a staggering $\$77.16$ per task. `GPT-5 Pro` is much more efficient, costing $\$7.14$ per task but only solving $18.3\%$ of tasks. Many other frontier models -- Claude, Grok, and Deepseek can't even crack $20\%$. In contrast, humans [get](https://arcprize.org/leaderboard) $100\%$ of questions right. That's why there exists a $\$1,000,000$ [competition](https://arcprize.org/competitions/2025/) to open source a solution to ARC-AGI-2. It's that difficult.""")
    return


@app.cell
def _(mo):
    mo.md(r"""# The HRM Drama""")
    return


@app.cell
def _(mo):
    mo.md(r"""In July, HRM was released. It is a fascinating model, inspired by the human brain with "slow" and "fast" loops of computation. It gained a lot of attention for it's amazing performance on ARC-AGI-1 despite its tiny size of 27M parameters.""")
    return


@app.cell
def _(mo):
    hrm_performance_image = mo.image(
        mo.notebook_dir() / "public/images/hrm_arc_agi.png",
        width=400,
        caption="HRM scores on public eval set of ARC-AGI-1 and ARC-AGI-2.",
        style={"display": "block", "margin": "0 auto"}
    )
    hrm_performance_image
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    HRM scored 40.3% on ARC-AGI-1 while SOTA models like o3-mini-high and Claude-3.7-8k scored 34.5%, and 21.2% respectively (back in July 2025). It beat Anthropic's best model by nearly ~2x! Similarly, it outperformed o3-mini-high and Claude-3.7-8k on ARC-AGI-2, but be warned that the ARC-AGI-2 the scores are so low that they are more much suspectable to noise.

    The results almost seemed to be too good to be true. How can a tiny 27M parameter model from a small lab be crushing some of the world's best models, at a fraction of their size?

    Turns out, HRM trained on test:
    """
    )
    return


@app.cell
def _(mo):
    hrm_train_on_eval_image = mo.image(
        mo.notebook_dir() / "public/images/hrm_train_on_eval_screenshot.png",
        width=600,
        caption="Screenshot of HRM paper showing that HRM trained on the public eval set of ARC-AGI-1.",
        style={"display": "block", "margin": "0 auto"}
    )

    hrm_train_on_eval_image
    return


@app.cell
def _(mo):
    mo.md(
        rf"""
    In their paper, the HRM authors admitted to showing the model "example pairs in the training and the **evaluation** sets". The evaluation set here refers to the public eval set of ARC-AGI-1! This sounds like training on test!

    On github, the HRM authors clarified that they only trained on the *examples* of the public eval set, not the *questions* of the public eval set. This "contraversy" set AI twitter on fire [[1](https://x.com/Dorialexander/status/1951954826545238181), [2](https://github.com/sapientinc/HRM/issues/18), [3](https://github.com/sapientinc/HRM/issues/1) [4](https://github.com/sapientinc/HRM/pull/22) [5](https://x.com/b_arbaretier/status/1951701328754852020)] ! Does this actually count as "training on test"? On one hand, you can never train on the data used to measure model perfomance. On the other hand, they never actually trained on the the questions used to measure model performance, just the examples associated with them.

    **What exactly is the difference between training on *examples* VS *questions* in ARC-AGI-1?**

    Consider a task from the public *eval* set, not the train set, of ARC-AGI-1:
    """
    )
    return


@app.cell
def _(display_task, ds):
    display_task(ds, "eval", 0)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    This task has two example pairs (left of the white line) and one test question (right of the white line). HRM was trained only on the examples—it never saw the actual test questions. Although the examples and questions come from the same distribution, the model still has to solve questions it's never encountered before. Since this is from the *eval* set, not the train set, we're essentially asking: you've seen these examples, now can you solve this closely-related but unseen problem?

    Is training on the eval set examples cheating? Apparently not. The ARC-AGI organizers accepted HRM's submision and the concsensus on [Twitter](https://x.com/Dorialexander/status/1951954826545238181) was that it's actually completely allowed.

    But buried in a GitHub thread, HRM's lead author, Guan Wang, made an offhand comment that caught my attention:
    > "If there were genuine 100% data leakage - then model should have very close to 100% performance (perfect memorization)." -   [Guan Wang](https://github.com/sapientinc/HRM/issues/1#issuecomment-3113214308)

    That line stuck with me. If partial leakage gets you $40.3\%$, what happens with *complete* leakage? If we train on the actual test questions, not just test examples, can we hit $100\%$? Can we do it with even fewer parameters than HRM (27M) or TRM (7M)? And can we do it on the more challenging ARC-AGI-2 instead of ARC-AGI-1? How far can we push this?
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Chaos Theory

    > "Chaos is what killed the dinosaurs, darling." - Joss Whedon
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    My goal was simple: create the tiniest possible model that achieves perfect performance on ARC-AGI-2 by blatantly training on the public eval set, both the examples and questions. We would deviate from HRM's acceptable approach (training on just the examples of the public eval set) and enter the morally dubious territory of training on the examples *and questions* of the public eval set.

    Now, the obvious approach would be to build a dictionary - just map each input directly to its corresponding output. But that's boring and lookup tables aren't nice mathematical functions. They're discrete, discontinuous, and definitely not differentiable. We need something else, something more elegant and interesting. To do that, we are going to take a brief detour into the world of chaos theory.

    > Note: Steven Piantadosi pioneered this technique in [One parameter is always enough](https://colala.berkeley.edu/papers/piantadosi2018one.pdf). Yet, I first heard of this technique through Laurent Boué's paper [Real numbers, data science and chaos: How to fit any dataset with a single parameter](https://arxiv.org/abs/1904.12320). This paper is really a gem due its sheer creativity.

    In chaos theory, the dyadic map $\mathcal{D}$ is a simple one-dimensional chaotic system defined as

    $$
    \begin{align}
    \mathcal{D}(a)
    &=
    (2a) \bmod 1
    &
    \mathcal{D}: [0, 1] \to [0, 1].
    \tag{2}
    \end{align}
    $$

    It takes in any number between 0 and 1, doubles it, and throws away the whole number part, leaving just the fraction. That's it.
    """
    )
    return


@app.function
def D(a): return (2 * a) % 1


@app.cell
def _(np, plt):
    def _():
        a_values = np.linspace(0, 1, 100)
        fig, ax = plt.subplots()
        ax.scatter(a_values, D(a_values), label="Dyadic", s=2)
        ax.set_xlabel(r"$a$")
        ax.set_ylabel(r"$\mathcal{D}(a)$")
        ax.set_title("Dyadic Map")
        ax.legend()
        # plt.show()
        return fig

    _()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    In chaos theory, we often study the orbit or trajectory of a chaotic system, the sequence generated by applying the chaotic map to itself over and over again. Starting with some number $a$, we apply our map to get $\mathcal{D}(a)$, and again to get $\mathcal{D}(\mathcal{D}(a))$, and so on and so forth. Let

    $$
    \begin{align*}
    \mathcal{D}^k(a)
    & :=
    \underbrace{(D \circ ... \circ D)}_{k}(a) = (2^k a) \mod 1
    \tag{3}
    \end{align*}
    $$

    mean we apply the dyadic map $k$ times to $a$. What does the orbit $(a, \mathcal{D}^1(a), \mathcal{D}^2(a), \mathcal{D}^3(a), \mathcal{D}^4(a), \mathcal{D}^5(a))$ look like?
    """
    )
    return


@app.function
def dyadic_orbit(a_L, k):
    orbits = [a_L]
    for _ in range(k):
        orbits.append(D(orbits[-1]))
    return orbits


@app.cell
def _():
    dyadic_orbit1 = dyadic_orbit(0.5, 5)
    return


@app.cell
def _():
    dyadic_orbit2 = dyadic_orbit(1/3, 5)
    return


@app.cell
def _():
    dyadic_orbit3 = dyadic_orbit(0.431, 5)
    return (dyadic_orbit3,)


@app.cell
def _(decimal_to_binary, dyadic_orbit3):
    dyadic_orbit3_binary = [decimal_to_binary(x, len(dyadic_orbit3)-i)[0] for i, x in enumerate(dyadic_orbit3)]
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    * If $a = 0.5$, the orbit is $(0.5, 0.0, 0.0, 0.0, 0.0, 0.0)$.
    * If $a = 1/3$, the orbit is $(0.333, 0.667, 0.333, 0.667, 0.333, 0.667,)$
    * If $a = 0.431$, the orbit is $(0.431, 0.862, 0.724, 0.448, 0.897, 0.792)$

    One orbit seems to end in all zeros, another bounces back and forth between $0.333$ and $0.667$, and a third seems to have no pattern at all. On the surface, these orbits do not have much in common. But if we take a closer look, they all share the same underlying pattern.

    Let's revisit the third orbit for $a = 0.431$ but this time we will analyze its binary representation:

    | Iterations | Decimal | Binary | Observation |
    |------------|------------------------|----------------------|-------------|
    | 0 | $a = 0.431$ | $\text{bin}(a) = 0.011011...$ | Original number |
    | 1 | $D^1(a) = 0.862$ | $\text{bin}(D^1(a)) = 0.11011...$ | First bit of $a$ $(0)$ removed |
    | 2 | $D^2(a) = 0.724$ | $\text{bin}(D^2(a)) = 0.1011...$ | First two bits of $a$ $(01)$ removed |
    | 3 | $D^3(a) = 0.448$ | $\text{bin}(D^3(a)) = 0.011...$ | First three bits of $a$ $(011)$ removed |
    | 4 | $D^4(a) = 0.897$ | $\text{bin}(D^4(a)) = 0.11...$ | First four bits of $a$ $(0110)$ removed |
    | 5 | $D^5(a) = 0.792$ | $\text{bin}(D^5(a)) = 0.1...$ | First four bits of $a$ $(01101)$ removed |

    Looking at the Binary column, we see that **every time we apply the dyadic map, the most significant bit is removed**! We start off with $0.011011$, and then applying $\mathcal{D}$ once removes the leftmost $0$ to get $0.11011$, and applying $\mathcal{D}$ another time removes the leftmost $1$ to get $0.1011$. Although the orbit appears irregular in its decimal representation, a clear pattern emerges from the binary representation.

    What is going on here?

    Each time we call $D(a) = (2a) \mod 1$, we double and truncate $a$. The doubling shifts every binary digit one place to the left and the truncation throws away whatever digit lands in the one's place. In other words, each application of $\mathcal{D}$ peels off the first binary digit and throws it away. **If we apply the dyadic map $k$ times, we remove the first $k$ bits of $a$.**

    We can see this process also holds for our other orbits:

    * If $a = 0.5$, we get the orbit $(0.5, 0.0, 0.0, 0.0, 0.0, 0.0)$ because $\text{bin}(a) = 0.100000...$ and after discarding the first bit, which is a $1$, we are left with all zeros.
    * If $a = 1/3$, we get the orbit $(0.333, 0.667, 0.333, 0.667, 0.333, 0.667)$ because $\text{bin}(a) = 0.010101...$, an infinite sequence of bits alternating between $1$ and $0$. When the bits start with a 0, we get $0.010101...$ which is $1/3 = 0.333$ in decimal. And when the bits start with a $1$, $0.10101...$, we get $2/3 = 0.667$ in decimal.

    Remarkably, these orbits are all governed by the same rule: remove one bit of information every time the dyadic map is applied. As each application of $\mathcal{D}$ removes another bit, this moves us deeper into the less significant digits of our original number -- the digits that are most sensitive to noise and measurement errors. A tiny change in $a$ due to noise, affecting the least significant bits of $a$, would eventually bubble up to the surface and completely change the orbit. That's why this system is so chaotic -- it is incredibly sensitive to even the smallest changes in the initial value $a$.

    (Note: we always compute the dyadic map on *decimal* numbers, not binary numbers; however, conceptually it is helpful to think about the binary representations of the orbit.)
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # The Dyadic Map As An ML Model
    > "When I grow up, I'm going to be a real ~~boy~~ ML Model" - the Dyadic Map if it were staring in Pinacoi
    """
    )
    return


@app.cell
def _():
    p_ = 6
    return (p_,)


@app.cell
def _(decimal_to_binary, p_):
    # initalize alpha
    b1 = decimal_to_binary(0.5, p_)[0]
    b2 = decimal_to_binary(1/3, p_)[0]
    b3 = decimal_to_binary(0.43085467085, p_)[0]
    b = ''.join([b1, b2, b3])
    print(f'{b1=}\n{b2=}\n{b3=}\n{b=}')
    return (b,)


@app.cell
def _(b, binary_to_decimal, decimal_to_binary, p_):
    alpha0_dec = binary_to_decimal(b)
    alpha0_bin = decimal_to_binary(alpha0_dec, 18)[0]
    b0_pred_bin = decimal_to_binary(alpha0_dec, p_)[0]
    x0_pred_dec = binary_to_decimal(b0_pred_bin)
    print(f'{alpha0_dec=}\n{alpha0_bin=}\nbin(alpha)[0:6]={b0_pred_bin}\nx^_0=dec(bin(alpha)[0:6])={x0_pred_dec}')
    return (alpha0_dec,)


@app.cell
def _(alpha0_dec, binary_to_decimal, decimal_to_binary, p_):
    alpha1_dec = dyadic_orbit(alpha0_dec, p_)[-1]
    alpha1_bin = decimal_to_binary(alpha1_dec, 18-p_)[0]
    b1_pred_bin = decimal_to_binary(alpha1_dec, p_)[0]
    x1_pred_dec = binary_to_decimal(b1_pred_bin)
    print(f'{alpha1_dec=}\n{alpha1_bin=}\nbin(D^6(alpha))[0:6]={b1_pred_bin}\nx^_1=dec(bin(D^6(alpha))[0:6])={x1_pred_dec}')
    return (alpha1_dec,)


@app.cell
def _(alpha1_dec, binary_to_decimal, decimal_to_binary, p_):
    alpha2_dec = dyadic_orbit(alpha1_dec, p_)[-1]
    alpha2_bin = decimal_to_binary(alpha2_dec, 18-2*p_)[0]
    b2_pred_bin = decimal_to_binary(alpha2_dec, p_)[0]
    x2_pred_dec = binary_to_decimal(b2_pred_bin)
    print(f'{alpha2_dec=}\n{alpha2_bin=}\nbin(D^12(alpha))[0:6]={b2_pred_bin}\nx^_2=dec(bin(D^12(alpha))[0:6])={x2_pred_dec}')
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    We've discovered something remarkable: each application of $\mathcal{D}$ peels away exactly one bit. But here's the question: if the dyadic map can systematically extract a number's bits, is it possible to put information in those bits in the first place? **What if we encode our dataset into a number's bits (`model.fit`) and then use the dyadic map as the core of a predictive model, extracting out the answer bit by bit (`model.predict`)?** In other words, can we turn the dyadic map into an ML model?

    Suppose our dataset contains the three numbers we saw before

    $$
    \mathcal{X}
    =
    \{x_0, x_1, x_2\}
    =
    \{0.5, 1/3,  0.431\}.
    $$

    Let's convert each number to binary and look at the first $p=6$ binary digits for simplicity:

    $$
    \mathcal{B}
    =
    \{b_0, b_1, b_2\}
    =
    \{ \text{bin}_6(x_0), \text{bin}_6(x_1), \text{bin}_6(x_2)\}
    =
    \{0.100000, 0.010101, 0.011011\}
    $$

    where the function $b_i = \text{bin}_p(x_i)$ converts decimal numbers to $p$-bit binary numbers. Now comes the clever part: we glue these binary strings together, end to end:

    $$
    b
    =
    0.
    \underbrace{100000}_{b_0}
    \underbrace{010101}_{b_1}
    \underbrace{011011}_{b_2}
    $$

    and convert this binary string back to decimal

    $$
    \alpha = \text{dec}(b) = 0.50522994995117188
    $$

    The number $\alpha$ is carefully engineered so that it is a decimal number whose bits contain our entire dataset's binary representation. That's right: **we've just compressed our entire dataset into a single decimal number!** We only have one parameter, not billions here! This is a very simple, stupid version of $\alpha = \text{model.fit}(\mathcal{X})$.

    But here's the question: given $\alpha$, how do we get our data $\mathcal{X}$ back out? How do we do $\tilde{x}_i = \text{model.predict}(\alpha)$? This is where the dyadic map becomes our extraction tool.

    *Step 1.* Trivially, we know the first 6 bits of $\alpha$ contains $b_0$.

    $$
    \begin{align*}
        \alpha
        &=
        0.50522994995117188
        \\
        \text{bin}(\alpha)
        &=
        0.\underbrace{100000}_{b_0}\underbrace{010101}_{b_1}\underbrace{011011}_{b_2}
        =
        0.100000010101011011
    \end{align*}
    $$

    So we'll just record the first $6$ bits of $\alpha$ to get $b_0$.

    $$
    \begin{align*}
        b_0
        &=
        \text{bin}(\alpha)_{0:6}
        =
        100000
    \end{align*}
    $$

    If we convert this number $b_0$ back to decimal, we'll recover our original data, up to the first $6$ digits of precision.

    $$
    \begin{align*}
        \tilde{x}_0
        &=
        \text{dec} ( b_0 )
        =
        0.500000
    \end{align*}
    $$

    Now from $\alpha$ we've extracted the prediction $\tilde{x}_0 = 0.500000$ which matches exactly the $0$th sample of our dataset $x_0 = 0.5$.

    *Step 2.* To predict the next number, $\tilde{x}_1$, remember that each application of $\mathcal{D}$ strips away the leftmost binary digit. So

    $$
    \begin{align*}
        D^6(\alpha)
        &=
        0.334716796875
    \end{align*}
    $$

    strips away the first $6$ bits of $\alpha$, which just removes $b_0$, and leaves us with $b_1, b_2$

    $$
    \begin{align*}
        \text{bin}(D^6(\alpha))
        &=
        0.\underbrace{\hspace{1cm}}_{b_0}\underbrace{010101}_{b_1}\underbrace{011011}_{b_2}
        =
        0.010101011011
    \end{align*}
    $$

    Like before, we'll then record the first $6$ bits of $D^6(\alpha)$ to get $b_1$

    $$
    \begin{align*}
        b_1
        &=
        \text{bin}(D^6(\alpha))_{0:6}
        =
        010101
    \end{align*}
    $$

    and convert $b_1$ back to decimal to get $\tilde{x}_1$.

    $$
    \begin{align*}
        \tilde{x}_1
        &=
        \text{dec} (b_1)
        =
        0.328125
    \end{align*}
    $$

    Here our prediction $\tilde{x}_1 = 0.328125$ is slightly off from the true value $x_1 = 1/3$ due to the limits of $6$-bit precision. If we'd have more digits of precision and increase $p$, $\tilde{x}_1$ would be closer to $x_1$.

    *Step 3.* To get the next number, $b_2$, apply $\mathcal{D}$ another 6 times, $\mathcal{D}^{12}(\alpha)$, removing another 6 bits of $\alpha$, i.e. $b_1$, and leaving us with just $b_2$. Like before we'll then record the first $6$ bits of $D^{12}(\alpha)$ to get $b_2$ and convert that back to decimal to get $\tilde{x}_2$.


    $$
    \begin{align*}
        D^{12}(\alpha)
        &=
        0.421875
        \\
        \text{bin}(D^{12}(\alpha))
        &=
        0.\underbrace{\hspace{1cm}}_{b_0}\underbrace{\hspace{1cm}}_{b_1}\underbrace{011011}_{b_2}
        =
        0.011011
        \\
        b_2
        &=
        \text{bin}(D^{12}(\alpha))_{0:6}
        =
        011011
        \\
        \tilde{x}_2
        &=
        0.421875
    \end{align*}
    $$

    Notice again that our prediction $\tilde{x}_2 = 0.421875$ is slightly off from the true value $x_2 = 0.431$ due to the limitations of $6$-bit precision.


    These 3 steps are summerized in the table below.

    | Iteration $i$ |$ip$ bits removed | $\mathcal{D}^{ip}(\alpha)$ in decimal | $\mathcal{D}^{ip}(\alpha)$ in binary | $b_i$, the first $p=6$ bits of $\mathcal{D}^{ip}(\alpha)$ in binary |  $\tilde{x}_i$, the first $p=6$ bits of $\mathcal{D}^{ip}(\alpha)$ in decimal|
    |------------|------------------------|----------------------|-------------|-------------|-------------|
    | $0$ | $0 \cdot 6 = 0$ | $\alpha = 0.50522994995117188$ | $\text{bin}(\alpha) = 0.\underbrace{100000}_{b_0}\underbrace{010101}_{b_1}\underbrace{011011}_{b_2}$ | $b_0 = 010101$ | $\tilde{x}_0 = 0.500000$|
    | $1$ | $1 \cdot 6 = 6$ | $\mathcal{D}^6(\alpha) = 0.33471679687500000$ | $\text{bin}(D^6(\alpha)) = 0.\underbrace{\hspace{1cm}}_{b_0}\underbrace{010101}_{b_1}\underbrace{011011}_{b_2}$ | $b_1 = 010101$| $\tilde{x}_1 = 0.328125$|
    | $2$ | $2 \cdot 6 = 12$ | $\mathcal{D}^{12}(\alpha) = 0.42187500000000000$ | $\text{bin}(D^{12}(\alpha)) = 0.\underbrace{\hspace{1cm}}_{b0}\underbrace{\hspace{1cm}}_{b1}\underbrace{011011}_{b_2}$ | $b_2 = 011011$| $\tilde{x}_2 = 0.421875$|

    In decimal, we go from $\alpha = 0.50522994995117188$ to $\mathcal{D}^6(\alpha) = 0.33471679687500000$ and then to $\mathcal{D}^{12}(\alpha) = 0.42187500000000000$. This pattern looks completely nonsensical. However, looking at the binary representation reveals that we are shifitng bits and extrating numbers with superb precision. This is anything but nonsensical. (Recall that we only every peform computation on the decimal numbers, never directly on their binary representation.)

    Think about what we've accomplished here. We just showed that you can take a dataset compress it down to a single real number, $\alpha$. Then, using nothing more than repeated doubling and truncation via $\mathcal{D}$, we can perfectly recover every data point in binary $\tilde{x}_0, \tilde{x}_1, \tilde{x}_2$ up to $p$ bits of precision. The chaotic dynamics of the dyadic map, which seemed like a nuisance, turns out to be the precise mechanism we need to systematically access that information.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    The algorithm itself is deceptively simple once you see the pattern:

    > **Encoding Algorithm:**
    > Given a dataset $\mathcal{X} = \{x_0, ..., x_{n-1}\}$ where $x_i \in [0, 1]$, encode the dataset into $\alpha$:
    >
    > 1. Convert each number to binary with $p$ bits of precision $b_i = \text{bin}_p(x_i)$ for $i=0, ..., n-1$
    > 2. Concatenate into a single binary string $b = b_0 \oplus  ... \oplus b_{n-1}$
    > 3. Convert to decimal $\alpha = \text{dec}(b)$


    The result is a single, decimal, scalar number $\alpha$ with $np$ bits of precision that contains our entire dataset. We can now discard $\mathcal{X}$ entirely.

    > **Decoding Algorithm:**
    > Given sample index $i \in \{0, ..., n-1\}$ and the encoded number $\alpha$, recover sample $\tilde{x_i}$:
    >
    > 1. Apply the dyadic map $\mathcal{D}$ exactly $ip$ times $\tilde{x}'_i = \mathcal{D}^{ip}(\alpha) = (2^{ip} \alpha) \mod 1$ 
    > 2. Extract the first $p$ bits of $\tilde{x}'_i$'s binary representation $b_i = \text{bin}_p(\tilde{x}'_i)$
    > 3. Covert to decimal $\tilde{x}_i = \text{dec}(b_i)$


    Mathematically, we can express these two algorithms with an encoder function $g: [0, 1]^n \to [0, 1]$ that compresses the dataset and a decoder function $f: \overbrace{[0, 1]}^{\alpha} \times \overbrace{\mathbb{Z}_+}^{p} \times \overbrace{[n]}^i \to [0, 1]$ that extracts individual data points:

    $$
    \begin{align*}
    \alpha
    &=
    g(p, \mathcal{X}) := \text{dec} \Big( \bigoplus_{x_i \in \mathcal{X}} \text{bin}_p(x_i) \Big)
    \tag{4}
    \\
    \tilde{x}_i
    &=
    f_{\alpha, p}(i) := \text{dec} \Big( \text{bin}_p \Big( \mathcal{D}^{ip}(\alpha) \Big) \Big)
    \end{align*}
    $$

    where $\oplus$ means concatenation.

    The precision parameter $p$ controls the trade-off between accuracy and storage efficiency. The larger $p$ is, the more accurately our encoding, but the more storage it takes up. Our error bound is

    $$
    |\tilde{x}_i - x_i | < \frac{1}{2^p}
    $$

    because we don't encode anything after the first $p$ bits of precision.

    What makes this profound is the realization that we're not really "learning" anything in any conventional sense. We're encoding it directly into the bits of a real number, exploiting it's infinite precision, and then using the dyadic map to navigate through that number and extract exactly what we need, when we need it.

    From this perspective, the dyadic map resembles a classical ML model where the encoder $g$ acts as `model.fit()` and the decoder $f$ acts as `model.predict()`.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Applying Some Makeup

    > "You don’t want to overdo it with too much makeup" - Heidi Klum
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    How do we go from the ugly, discontinuous decoder function

    $$
    f_{\alpha,p}(i) := \text{dec} \Big( \text{bin}_p \Big( \mathcal{D}^{ip}(\alpha) \Big) \Big)
    $$

    to that beautiful function I promised you at the start of the blog

    $$
    f_{\alpha, p}(x)
    =
    \sin^2 \Big(
        2^{i p} \arcsin^2(\sqrt{\alpha})
    \Big)
    ?
    $$

    In this section we will "apply makeup" to the first function to get it looking a bit closer to the second function. We will keep the same core logic but make the function more ascetically pleasing. To do this, we will need another one-dimensional chaotic system, the [logistic map](https://en.wikipedia.org/wiki/Logistic_map) at $r=4$ on the unit interval:

    $$
    \begin{align*}
    \mathcal{L}(a_L)
    &=
    4 a_L (1 - a_L)
    &
    \mathcal{L}: [0, 1] \to [0, 1]
    \tag{6}
    \end{align*}
    $$

    which seems quite different than the familiar dyadic map

    $$
    \begin{align*}
    \mathcal{D}(a_D)
    &=
    (2 a_D) \mod 1
    &
    \mathcal{D}: [0, 1] \to [0, 1]
    \end{align*}
    $$

    One is a bit-shifting operation, the other is a smooth parabola that ecologists use to model population growth. (Note: previously $a$ was the input to the dyadic map but from now on $a_D$ will be the input to the dyadic map to differentiate it from $a_L$, the input to the logistic map.)
    """
    )
    return


@app.function
def L(a): return 4 * a * (1 - a)


@app.cell
def _(np, plt):
    def _():
        a_values = np.linspace(0, 1, 100)

        fig, ax = plt.subplots()
        ax.scatter(a_values, D(a_values), label="Dyadic", s=2)
        ax.scatter(a_values, L(a_values), label="Logistic", s=2)
        ax.set_xlabel(r"$a$")
        ax.set_ylabel(r"$\mathcal{D}(a)$ or $\mathcal{L}(a)$")
        ax.set_title("Logistic vs Dyadic Map")
        ax.legend()
        # plt.show()
        return fig


    _()
    return


@app.function
def logistic_orbit(a_L, k):
    orbits = [a_L]
    for _ in range(k):
        orbits.append(L(orbits[-1]))
    return orbits


@app.cell
def _():
    logistic_orbit(0.5, 5)
    return


@app.cell
def _():
    logistic_orbit(1/3, 5)
    return


@app.cell
def _():
    logistic_orbit(0.431, 5)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    What does the logistic orbit $(a_L, \mathcal{L}^1(a_L), \mathcal{L}^2(a_L), \mathcal{L}^3(a_L), \mathcal{L}^4(a_L), \mathcal{L}^5(a_L))$ look like? Similar or different to the dyadic orbit $(a_D, \mathcal{D}^1(a_D), \mathcal{D}^2(a_D), \mathcal{D}^3(a_D), \mathcal{D}^4(a_D), \mathcal{D}^5(a_D))$?

    * If $a_L = a_D = 0.5$, the logistic orbit is $(0.5, 1.0, 0.0, 0.0, 0.0, 0.0)$ while the dyadic orbit is $(0.5, 0.0, 0.0, 0.0, 0.0, 0.0)$. Although the dyadic orbit is just all zeros, the logistic orbit actually has $1.0$ as the second number in the orbit.
    * If $a_L = 1/3$, the logistic orbit is $(0.333, 0.888, 0.395, 0.956, 0.168, 0.560)$ while the dyadic orbit is $(0.333, 0.667, 0.333, 0.667, 0.333, 0.667)$. While the dyadic orbit repeats in a simple pattern, the logistic orbit is seemingly patternless.
    * If $a_L = 0.43085467085$, the logistic orbit is $(0.431, 0.981, 0.075, 0.277, 0.800, 0.639)$ while the dyadic orbit is $(0.431, 0.862, 0.724, 0.448, 0.897, 0.792)$. Both orbits here seem totally random and chaotic, but each in their own way.

    The logistic and dyadic maps create orbits that look nothing alike!

    However, [topological conjugacy](https://en.wikipedia.org/wiki/Topological_conjugacy) tells us these two maps are *actually* the same. Not similar. Not analgous. The same. They have identical orbits, the exact same chaotic trajectories, simply expressed in different coordinates. The logistic map, for all its smooth curves and elegant form, is actually doing discrete binary operations under the hood, just like the dyadic map (and vice versa). Formally, two functions are topologically conjugate if there exists a homeomorphism, fancy talk for a change of coordinates, that perfectly takes you from one map to another. The change of coordinates here is

    $$
    \begin{align*}
    a_L = \phi(a_D) &= \sin^2(2 \pi a_D)
    &
    \phi: [0, 1] -> [0, 1]
    \tag{7}
    \\
    a_D
    =
    \phi^{-1}(a_L)
    &=
    \frac{1}{2 \pi} \arcsin (\sqrt{a_L})
    &
    \phi^{-1}: [0, 1] -> [0, 1]
    \tag{8}
    \end{align*}
    $$

    We can map any $a_L$ to an $a_D$ and any $a_D$ to an $a_L$.
    """
    )
    return


@app.cell
def _(np, plt):
    def _():
        a_values = np.linspace(0, 1, 100)
        def phi(a): return np.sin(2 * np.pi * a) ** 2
        def phi_inverse(a): return np.arcsin(np.sqrt(a)) / (2.0 * np.pi)

        fig, ax = plt.subplots()
        ax.scatter(a_values, phi(a_values), label="phi", s=2, c="g")
        ax.scatter(a_values, phi_inverse(a_values), label="phi inverse", s=2, c="r")
        ax.set_xlabel("a")
        ax.set_ylabel("output")
        ax.set_title("phi and phi inverse")
        ax.legend()
        # plt.show()
        return fig

    _()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Looking at the plot, the function $\phi$ has a period of 1, meaning it repeats the same values every time $a_D$ increases by $1$. This periodicity is crucial because it allows us to drop the modulo operation from the dyadic map $\mathcal{D}(a_D) = (2 a_D) \mod 1$ when transforming from the dyadic space to the logistic space. Formally,

    $$
    \begin{align*}
    \phi(a_D \mod 1) = \phi(a_D)
    \tag{9}
    \end{align*}
    $$

    which will be important later on. To go back and forth between the dyadic and logistic maps, we apply $\phi$ to the output $\mathcal{D}$ and get $\mathcal{L}$; we can also apply $\phi^{-1}$ to the input $a_L$ to get $\mathcal{D}$. Mathemtically,

    $$
    \begin{align*}
    \mathcal{L}(a_L)
    &=
    \phi(\mathcal{D}(a_D))
    \\
    \mathcal{D}(a_D)
    &=
    \mathcal{L}(\phi^{-1}(a_L))
    \end{align*}
    $$

    Here $\phi$ takes us to the logistic space and $\phi^{-1}$ takes us back to the dyadic space. This is astonishing! $\phi$ is just a sin wave squared and with a period of one. It's inverse, $\phi^{-1}$ is even weirder looking with an $\arcsin$. But somehow these functions allow us to bridge the two maps $\mathcal{D}$ and $\mathcal{L}$!

    Moreover, $\phi$ and $\phi^{-1}$ perfectly relate *every* single point in the infinite orbits of $\mathcal{D}$ and $\mathcal{L}$:

    $$
    (a_D, \mathcal{D}^1(a_D), \mathcal{D}^2(a_D), ...) = (a_L, \mathcal{L}^1(\phi^{-1}((a_L)), \mathcal{L}^2(\phi^{-1}((a_L)), ...)
    $$

    or it can be expressed as

    $$
    (a_L, \mathcal{L}^1(a_L), \mathcal{L}^2(a_L), ...) = (a_D, \phi(\mathcal{D}^1(a_D)), \phi(\mathcal{D}^2(a_D)), ...)
    $$

    depending on if we want to be natively using the coordinate system of $\mathcal{D}$ or $\mathcal{L}$. What appears as chaos in one coordinate system manifests as the exact same chaos in the other, *no matter how many iterations we apply*. Mathematically, this suggests something stronger:

    $$
    \begin{align*}
    \mathcal{L}^k(a_L)
    &=
    \phi(\mathcal{D}^k(a_D))
    \tag{10}
    \\
    \mathcal{D}^k(a_D)
    &=
    \mathcal{L}^k(\phi^{-1}(a_L))
    \tag{11}
    \end{align*}
    $$

    Think of these two orbits existing in parallel universes with $\phi$ and $\phi^{-1}$ acting as the bridges between $\mathcal{D}$ and $\mathcal{L}$.
    """
    )
    return


@app.cell
def _(mo):
    topological_conjugacy_image = mo.image(
        mo.notebook_dir() / "public/images/topological_conjugacy.png",
        width=400,
        caption="Topological conjugacy between the dyadic and logistic map.",
        style={"display": "block", "margin": "0 auto"}
    )
    return (topological_conjugacy_image,)


@app.cell
def _(mo, topological_conjugacy_image):
    mo.md(f"""{topological_conjugacy_image}""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Revisting the dyadic and logistic orbits for $a_D = a_L = 0.431$, we can take the dyadic orbit $(0.431, 0.862, 0.724, 0.448, 0.897, 0.792)$ and apply $\phi$ to every element, giving us $(0.431, 0.981, 0.075, 0.277, 0.800, 0.639)$  -- which is exactly the logistic orbit (eqn. (10))! Similarly, we can take the logistic orbit $(0.431, 0.981, 0.075, 0.277, 0.800, 0.639)$, apply $\phi^{-1}$
    to get the dyadic orbit $(0.431, 0.862, 0.724, 0.448, 0.897, 0.792)$

    We see that although both these orbits look completly unrelated, these two orbits are perfectly connected to one another through $\phi$ and $\phi^{-1}$.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    **Can we use the topological conjugacy of $\mathcal{D}$ and $\mathcal{L}$ as makeup?**

    While $\mathcal{D}$ is ugly and discontinuous, $\mathcal{L}$ is smooth and differentiable. We can use the logistic map as "makeup" to hide the crude dyadic operations. We want our decoder to use $\mathcal{L}$ instead of $\mathcal{D}$. But for the encoder to glue together the bits of our dataset, we needs to be in the dyadic space so our clever bit manipulations will still work out. Here's the strategy:

    1. Encoder: Work in dyadic space where bit manipulation works (use $\phi$) but output parameter in logistic space (use $\phi^{-1}$)
    2. Decoder: Work entirely in smooth logistic space using the conjugacy relationship

    This gives us two new beautiful encoder/decoder algorithms where the main changes are bolded:

    > **Encoding Algorithm:**
    > Given a dataset $\mathcal{X} = \{x_0, ..., x_n\}$ where $x_i \in [0, 1]$, encode the dataset into $a_L$:
    >
    > 1. ***Transform data to dyadic coordinates: $z_i = \phi^{-1}(x_i) = \frac{1}{2 \pi} \arcsin⁡( x_i )$ for $i=1, ..., n$***
    > 2. Convert each transformed number to binary with $p$ bits of precision: $b_i = \text{bin}_p(z_i)$ for $i=1, ..., n$
    > 3. Concatenate into a single binary string $b = b_0 \oplus  ... \oplus b_n$
    > 4. Convert to decimal $a_D = \text{dec}(b)$
    > 5. ***Transform to logistic space: $\alpha = a_L = \phi(a_D) = \sin^2(2 \pi a_D)$***

    The result is a single, decimal, scalar number $\alpha$ with $np$ bits of precision that contains our entire dataset. We can now discard $\mathcal{X}$ entirely.

    > **Decoding Algorithm:**
    > Given sample index $i$ and the encoded number $\alpha$, recover sample $\tilde{x_i}$:
    >
    > 1. ***Apply the logistic map $\mathcal{L}$ exactly $ip$ times $\tilde{x}'_i = \mathcal{L}^{ip}(\alpha) = \sin^2 \Big(2^{i p} \arcsin^2(\sqrt{\alpha}) \Big)$***
    > 2. Extract the first $p$ bits of $\tilde{x}'_i$'s binary representation $b_i = \text{bin}_p(\tilde{x}'_i)$
    > 3. Covert to decimal $\tilde{x}_i = \text{dec}(b_i)$
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Mathematically, we can express this with a new and improved encoder $g$ and decoder $f$:

    $$
    \begin{align*}
    \alpha
    &=
    g(p, \mathcal{x}) := \phi \bigg( \text{dec} \Big( \bigoplus_{x \in \mathcal{X}} \text{bin}_p(\phi^{-1}(x)) \Big) \bigg)
    \\
    \tilde{x}_i
    &=
    f_{\alpha,p}(i)
    :=
    \text{dec} \Big( \text{bin}_p \Big( \mathcal{L}^{ip}(\alpha) \Big) \Big)
    =
    \text{dec} \Big( \text{bin}_p \Big( \sin^2 \Big(2^{ip} \arcsin(\sqrt{\alpha}) \Big) \Big) \Big)
    \end{align*}
    $$

    where $\oplus$ means concatenation. The decoder here is tantalizingly close to the function I promised at the start:

    $$
    f_{\alpha, p}(x)
    =
    \sin^2 \Big(
        2^{x p} \arcsin^2(\sqrt{\alpha})
    \Big)
    $$

    but is still wrapped with those pesky $\text{dec}$ and $\text{bin}_p$ operations. However, something profound has happened here. We've taken the crude, discontinuous dyadic map and transformed it into something smooth and differentiable. The logistic map doesn't *look* like it's doing binary operations, but underneath the elegant trigonometry, it's performing exactly the same bit manipulations as its topological coungant, the dyadic map. Indeed, the makeup looks pretty great!

    However, nothing is free. The cost of using the logistic map instead of the dyadic map is that our error is now $2 \pi$ times larger, $|\tilde{x}_i - x_i | < \frac{\pi}{2^{p-1}}$. (We get this $2 \pi$ factor by noting that the derivative of $\phi$ is bounded by $2 \pi$ and applying the mean-value theorem.)
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// details | How did we get $\mathcal{L}^{ip}(\alpha) = \sin^2 \Big(2^{i p} \arcsin^2(\sqrt{\alpha}) \Big)$?

    We just need to perform some simple algebraic manipulation with our equations:

    $$
    \begin{align*}
    \mathcal{L}^k(\alpha)
    &=
    \mathcal{L}^k(a_L)
    &
    \text{by $\alpha = a_L$}
    \\
    &=
    \phi(\mathcal{D}^k(a_D))
    &
    \text{by $(10)$}
    \\
    &=
    \phi((2^k a_D) \mod 1)
    &
    \text{by $(3)$}
    \\
    &=
    \phi(2^k a_D)
    &
    \text{by $(9)$}
    \\
    &=
    \sin^2(2 \pi \cdot (2^k a_D))
    &
    \text{by $(7)$}
    \\
    &=
    \sin^2 \bigg(2 \pi 2^k \Big( \frac{1}{2 \pi} \arcsin(\sqrt{a_L}) \Big) \bigg)
    &
    \text{by $(8)$}
    \\
    &=
    \sin^2 \Big(2^k \arcsin(\sqrt{a_L}) \Big)
    &
    \text{by simplification}
    \\
    &=
    \sin^2 \Big(2^k \arcsin(\sqrt{\alpha}) \Big)
    &
    \text{by $\alpha = a_L$}
    \end{align*}
    $$
    ///
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""# Code Implementation""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Now comes the moment of truth. We've built up all this beautiful theory about chaos and topological conjugacy, but can we actually code it up?

    If you've been paying attention, there is one crucial implementation detail we have to worry about. If our dataset $\mathcal{X}$ has $n$ samples, each encoded with $p$ bits, $\alpha$ will contain $np$ bits. This far exceeds the $32$ or $64$ bits that standard computers can handle. How do we even represent $\alpha$ numerically on a computer?

    Simple: we can use an arbitrary precision arithmetic library like [gmpy2](https://github.com/aleaxit/gmpy) to handle numbers with any precision we want. Instead of representing $\alpha$ as a regular Python float, we can just represent it as an gmpy2 float with $np$ bits.

    But gmpy2 does more than just let us represent impossibly large numbers. It also simplifies our decoder equation

    $$
    \begin{align*}
    f_{\alpha,p}(i)
    & :=
    \text{dec} \Big( \text{bin}_p \Big( \mathcal{L}^{ip}(\alpha) \Big) \Big)
    =
    \text{dec} \Big( \text{bin}_p \Big( \sin^2 \Big(2^{ip} \arcsin(\sqrt{\alpha}) \Big) \Big) \Big)
    \end{align*}
    $$

    In our code, we can set gmpy2's working precision to $np$ bits when we compute $\mathcal{L}^{ip}(\alpha)$. Then we simply change the precision to $p$ bits, and gmpy2 automatically gives us just the first $p$ bits we care about. With `gmpy2`, there is no need to explicitly convert $\mathcal{L}^{ip}(\alpha)$ to binary, extract the first $p$ bits, and then convert it back to decimal -- this is automatically taken care of for us. Therefore our decoder becomes even simpler:

    $$
    \begin{align*}
    f_{\alpha,p}(i)
    &=
    \mathcal{L}^{ip}(\alpha)
    =
    \sin^2 \Big(2^{ip} \arcsin(\sqrt{\alpha}) \Big)
    \tag{5}
    \end{align*}
    $$

    Usually translating elegant math equations into code turns beautiful theory into ugly, complicated messes—but surprisingly, leveraging gmpy2 had the opposite effect and actually made our decoder even simpler.

    In code our logistic decoder is:
    """
    )
    return


@app.cell
def _(display_fxn, logistic_decoder, mo):
    mo.md(rf"""{display_fxn(logistic_decoder)}""")
    return


@app.cell
def _(mo):
    mo.md(r"""Now, let's define some basic helper functions for $\text{bin}_p, \text{dec}, \phi, \phi^{-1}$. Note that we compute $\phi^{-1}$ in numpy but use `gmpy2` to compute $\phi$.""")
    return


@app.cell
def _(binary_to_decimal, decimal_to_binary, display_fxn, mo, phi, phi_inverse):
    mo.md(
        rf"""
    {display_fxn(binary_to_decimal)}

    {display_fxn(decimal_to_binary)}

    {display_fxn(phi)}

    {display_fxn(phi_inverse)}
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""Because `logistic_decoder_single` is so slow, we need a parallelized implemntation of the logistic map so our code can run in a reasonable amount of time.""")
    return


@app.cell
def _(display_fxn, logistic_decoder, mo):
    mo.md(rf"""{display_fxn(logistic_decoder)}""")
    return


@app.cell
def _(mo):
    mo.md(r"""Finally, we can define our one parameter model. Our `fit` method implements the encoder function and the `predict` method implements the deocder function.""")
    return


@app.cell
def _(OneParameterModel, display_fxn, mo):
    mo.md(rf"""{display_fxn(OneParameterModel)}""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    /// details | Can you walk me through what `ScalarModel` does line-by-line?


    Let's walk through this big chunk of code line-by-line.

    We initialize the model with the desired precision $p$ and the number of workers for running our decoder in parallel.
    ```py
    class OneParameterModel:
        def __init__(self, precision, workers=8):
            self.precision = precision # binary precision, not decimal precision, for a single number
            self.workers = workers
    ```

    Then we have the `fit` method which implements our encoder $g$.
    ```py
        @Timing("fit: ", enabled=VERBOSE)
        def fit(self, X, y):
    ```

    Previously we only encoded *unsupervised* datasets $D = (\mathcal{X})$ using $g(\mathcal{X})$. But we also need to be able to encode *supervised* datasets $D = (\mathcal{X}, \mathcal{Y})$ using $g(\mathcal{Y})$. In the supervised setting we encode the labels $g(\mathcal{Y})$ and not the data $g(\mathcal{X})$ because we really care about predicting the output label, not the input data. This is why all of our encoding and decoding will be done on `y` and not `X`. However, if the dataset is *unsupervised* and there is no `y`, we set `y = X` to get around this.
    ```py
            # if the dataset is unsupervised, treat the data X like the labels y
            if y is None: y = X
    ```

    Next we determine the total precision from `y`, not `X`, which should be $np$ bits where $p$ is the precision we choose and $n$ is the number of elements in $y$:
    ```py
            self.y_shape = y.shape[1:] # pylint: disable=attribute-defined-outside-init
            self.total_precision = y.size * self.precision # pylint: disable=attribute-defined-outside-init
    ```

    We scale `y` to be in $[0, 1]$ because our encoder $g$ only works on values in the unit interval.
    ```py
            # scale labels to be in [0, 1]
            self.scaler = MinMaxScaler() # pylint: disable=attribute-defined-outside-init
            y_scaled = self.scaler.scale(y.flatten())
    ```
    Now we actually implement the five steps of our encoder $g$ using `gmpy2` to make sure all of our calculations use the correct precision. Step by step we: compute $\phi^{-1}$ on all labels, convert to binary, concatenate all binary strings, convert to a scalar decimal, and then apply $\phi$.
    ```py
            # compute alpha with arbitrary floating-point precision
            with gmpy2.context(precision=self.total_precision):

                # 1. compute φ^(-1) for all labels
                phi_inv_decimal_list = phi_inverse(y_scaled)
                # 2. convert to a binary
                phi_inv_binary_list = decimal_to_binary(phi_inv_decimal_list, self.precision)
                # 3. concatenate all binary strings together
                phi_inv_binary = ''.join(phi_inv_binary_list)
                if len(phi_inv_binary) != self.total_precision:
                    raise ValueError(f"Expected {self.total_precision} binary digits but got {len(phi_inv_binary)}.")

                # 4. convert to a scalar decimal
                phi_inv_scalar = binary_to_decimal(phi_inv_binary)
                # 5. apply φ to the scalar
                self.alpha = phi(phi_inv_scalar) # pylint: disable=attribute-defined-outside-init

                if VERBOSE >= 2: print(f'With {self.precision} digits of binary precision, alpha has {len(str(self.alpha))} digits of decimal precision.')
                if VERBOSE >= 3: print(f'{self.alpha=}')
            return self
    ```


    Now the `predict` method implement the decoder $f$. It takes in a list of indices `idxs` that we want to decode.
    ```py
        @Timing("predict: ", enabled=VERBOSE)
        def predict(self, idxs):
    ```
    When we defined the encoder $g$ and decoder $f$ functions, they only encoded/decoded scalars -- look at their mathemtical formulations. But what is our labels are vectors or matricies as they are in ARC-AGI-1? To handle this in `fit`, we called `y.flatten()` to flatten the vector or matrix into a long list of scalars. This way it still feels like we are only dealing with a list of scalars. In `predict` we do some fancy indicing to correct for the flattening.
    ```py
            y_size = np.array(self.y_shape).prod()
            full_idxs = (np.tile(np.arange(y_size), (len(idxs), 1)) + idxs[:, None] * y_size).flatten()
    ```
    Now with the corrected indicies, `full_idxs`, we can actually run the decoder implemented with the logistic map, not the dyadic map.
    ```py
            raw_pred = logistic_decoder(self.total_precision, self.alpha, self.precision, full_idxs)
    ```
    Lastly, we undo both the flattening and the unit interval scaling to get our outputted prediction.
    ```py
            return self.scaler.unscale(raw_pred).reshape((-1, *self.y_shape))
    ```
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Our goal is to use `OneParameterModel` for ARC-AGI-1. However, the ARC-AGI-1 dataset consists of matrices of integers $0$ through $9$. So we had to make three key changes to the encoder and decoder in `OneParameterModel`:

    1. **Data scaling.** ARC-AGI-1 uses integers 0-9, but our encoder needs values in [0,1]. We use a standard MinMaxScaler to squeeze the data into the right range.
    2. **Shape handling.** Our encoder works on datasets with scalar numbers, not matrices. Simple solution: flatten the matrices into long lists during encoding and then reshape back during decoding.
    3. **Supervised learning.** ARC-AGI-1 is a supervised learning problem with input-output pairs $(X,Y)$, but our encoder can only handle unsupervised datasets $(X)$. We simply encode the outputs $Y$ instead of the inputs $X$ because the outputs $Y$ are what we actually need to predict.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Let's try it our model out on ARC-AGI-1!

    We will use the public eval set from ARC-AGI-1 which has 400 tasks. We can ignore the example input-output pairs and only look at the question inputs-output pairs because we are only actually predictioning the question outputs given the question inputs.
    """
    )
    return


@app.cell
def _(ds, mo, process_arc_agi):
    X, y = process_arc_agi(ds)
    X, y = X[:10], y[:10]
    mo.show_code()
    return X, y


@app.cell
def _(mo):
    mo.md(r"""`X` contains 400 question inputs and `y` contains 400 question outputs. Each input and output is a 30 by 30 grid (list of lists) of integers between $0$ and $9$.""")
    return


@app.cell
def _(X, mo, y):
    with mo.redirect_stdout():
        print(f'{X.shape=} {y.shape=}')
    return


@app.cell
def _(mo):
    mo.md(r"""Now we are ready to "train" our model and compress our arc-agi-1 dataset into $\alpha$. For simplicity, we will train on the first 5 examples of arc-agi.""")
    return


@app.cell
def _(OneParameterModel, X, mo, y):
    p = 6
    model = OneParameterModel(p)
    model.fit(X, y)
    mo.show_code()
    return (model,)


@app.cell
def _(mo, model):
    mo.md(f"""```py\nmodel.alpha={str(model.alpha)[:10_000]}\n```""")
    return


@app.cell
def _(mo):
    mo.md(r"""Within a couple of seconds, we have learned $\alpha$! This wonderful, magical scalar number is the key to getting a perfect score on ARC-AGI-1. Watch:""")
    return


@app.cell
def _(model, np):
    X_idx = np.array([0])
    y_pred = model.predict(X_idx)
    return X_idx, y_pred


@app.cell
def _(X_idx, model, y, y_pred):
    model.verify(y_pred, y[X_idx])
    return


@app.cell
def _():
    # # import matplotlib.pyplot as plt
    # # import matplotlib.colors as colors
    # # import numpy as np
    # from matplotlib.colors import to_rgb

    # base_colors = ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
    #                '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25']
    # cmap2 = colors.ListedColormap(base_colors, N=256)


    # from matplotlib.colors import to_rgb

    # # def get_text_color(bg_color):
    # #     """White text on dark bg, black on light bg"""
    # #     r, g, b = to_rgb(bg_color)
    # #     brightness = 0.299*r + 0.587*g + 0.114*b
    # #     return 'white' if brightness < 0.5 else 'black'

    # def add_values_to_plot(ax, matrix, cmap, norm):
    #     """Add numerical values as text overlay on matrix plot"""
    #     height, width = matrix.shape
    #     fontsize = max(5, min(10, 80 / max(height, width)))

    #     for i in range(height):
    #         for j in range(width):
    #             value = matrix[i, j]

    #             # Format: show integer without decimal, float with 1 decimal
    #             if value == int(value):
    #                 text = f'{int(value)}'
    #             else:
    #                 text = f'{value:.1f}'

    #             # Get background color and choose contrasting text color
    #             cell_color = cmap(norm(value))
    #             text_color = get_text_color(cell_color)

    #             ax.text(j, i, text,
    #                    ha='center', va='center',
    #                    color=text_color,
    #                    fontsize=fontsize,
    #                    fontweight='bold')

    # def plot_matrix2(matrix, ax=None, title=None, vmin=None, vmax=None, grid_w=0.8, status=None, show_vals=False):
    #     if ax is None:
    #         fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    #         fig.patch.set_facecolor('#444444')
    #     else:
    #         fig = ax.get_figure()

    #     if vmin is None:
    #         vmin = matrix.min()
    #     if vmax is None:
    #         vmax = matrix.max()

    #     norm = colors.Normalize(vmin=vmin, vmax=vmax)
    #     ax.imshow(matrix, cmap=cmap2, norm=norm)

    #     ax.set_xticks([x - 0.5 for x in range(1 + matrix.shape[1])])
    #     ax.set_yticks([x - 0.5 for x in range(1 + matrix.shape[0])])
    #     ax.grid(True, which='both', color='#666666', linewidth=grid_w)
    #     ax.tick_params(axis='both', color='none', length=0)
    #     ax.set_xticklabels([])
    #     ax.set_yticklabels([])

    #     if show_vals:
    #         add_values_to_plot(ax, matrix, cmap2, norm)

    #     if title:
    #         ax.set_title(f'\n{title}', fontsize=12, color='#dddddd')

    #     if status:
    #         ax.text(1, 1.15, status[0],
    #                transform=ax.transAxes,
    #                ha='right', va='bottom',
    #                fontsize=10, fontweight='bold',
    #                color=status[1])

    #     return fig, ax

    # def plot_one2(ax, i, task, ex_or_q, in_or_out, w=0.8, vmin=None, vmax=None, show_vals=False):
    #     matrix = task[f"{ex_or_q}_{in_or_out}"][i]
    #     title = f'{ex_or_q.capitalize()} {i} {in_or_out[:-1].capitalize()}'

    #     if ex_or_q == 'question' and in_or_out == 'outputs':
    #         status = ('? PREDICT', '#FF4136')
    #     else:
    #         status = ('✓ GIVEN', '#2ECC40')

    #     plot_matrix2(matrix, ax=ax, title=title, vmin=vmin, vmax=vmax, grid_w=w, status=status, show_vals=show_vals)

    # def display_task2(ds, split, i, size=2.5, w=0.9, vmin=None, vmax=None, show_vals=False):
    #     task = ds[split][i]
    #     n_ex = len(task['example_inputs'])
    #     n_q = len(task['question_inputs'])

    #     # Auto-detect vmin/vmax if not provided
    #     if vmin is None or vmax is None:
    #         all_values = []

    #         for j in range(n_ex):
    #             all_values.extend(task['example_inputs'][j].flatten())
    #             all_values.extend(task['example_outputs'][j].flatten())

    #         for k in range(n_q):
    #             all_values.extend(task['question_inputs'][k].flatten())
    #             if task['question_outputs']:
    #                 all_values.extend(task['question_outputs'][k].flatten())

    #         if vmin is None:
    #             vmin = min(all_values)
    #         if vmax is None:
    #             vmax = max(all_values)

    #     # Create subplot grid
    #     total_cols = n_ex + n_q
    #     fig, axs = plt.subplots(2, total_cols, figsize=(size * total_cols, 2 * size))
    #     plt.suptitle(f'ARC-AGI-1 {split.capitalize()} Task #{i} (id={task["id"]})',
    #                  fontsize=16, fontweight='bold', y=1, color='#eeeeee')

    #     # Plot examples
    #     for j in range(n_ex):
    #         plot_one2(axs[0, j], j, task, 'example', 'inputs', w, vmin, vmax, show_vals)
    #         plot_one2(axs[1, j], j, task, 'example', 'outputs', w, vmin, vmax, show_vals)

    #     # Plot questions
    #     for k in range(n_q):
    #         plot_one2(axs[0, n_ex + k], k, task, 'question', 'inputs', w, vmin, vmax, show_vals)
    #         plot_one2(axs[1, n_ex + k], k, task, 'question', 'outputs', w, vmin, vmax, show_vals)

    #     # Add separator line
    #     axs[1, n_ex].set_xticklabels([])
    #     axs[1, n_ex].set_yticklabels([])
    #     axs[1, n_ex] = plt.figure(1).add_subplot(111)
    #     axs[1, n_ex].set_xlim([0, total_cols])
    #     axs[1, n_ex].plot([n_ex, n_ex], [0, 1], '-', linewidth=5, color='white')
    #     axs[1, n_ex].axis("off")

    #     # Style the figure
    #     fig.patch.set_linewidth(5)
    #     fig.patch.set_edgecolor('black')
    #     fig.patch.set_facecolor('#444444')
    #     plt.tight_layout(h_pad=3.0)

    #     return fig
    return


@app.cell
def _():
    # x_small, y_small = -1, -1
    # y_pred_plot = y_pred.squeeze()[:x_small, :y_small]
    # plot_matrix2(y_pred_plot, title="y_pred", vmin=0, vmax=9, show_vals=True)
    return


@app.cell
def _():
    # y_plot = y[X_idx].squeeze()[:x_small, :y_small]
    # plot_matrix2(y_plot, title="y", vmin=0, vmax=9, show_vals=True)
    return


@app.cell
def _():
    # import plotly.graph_objects as go
    # from plotly.subplots import make_subplots

    # COLORS = ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
    #           '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25']

    # def get_text_color(val):
    #     r, g, b = int(COLORS[val][1:3], 16), int(COLORS[val][3:5], 16), int(COLORS[val][5:7], 16)
    #     return 'white' if (0.299*r + 0.587*g + 0.114*b)/255 < 0.5 else 'black'

    # def plot_matrix(mat, fig=None, row=None, col=None, title=None, show_nums=False):
    #     mat = np.array(mat)
    #     h, w = mat.shape

    #     # Create cells and hovertext with numerical
    #     cells, annotations, hover_x, hover_y, hover_text = [], [], [], [], []
    #     for i in range(h):
    #         for j in range(w):
    #             v = mat[i, j]
    #             cells.append(dict(type='rect', x0=j, x1=j+1, y0=h-i-1, y1=h-i, fillcolor=COLORS[v], line=dict(color='#666', width=1)))
    #             hover_x.append(j+0.5)
    #             hover_y.append(h-i-0.5)
    #             hover_text.append(f'Value: {v}')
    #             if show_nums: annotations.append(dict(x=j+0.5, y=h-i-0.5, text=str(v), showarrow=False, font=dict(color=get_text_color(v), size=10)))
    
    #     # Annotate plot with example number "Ex 3." and matrix dimensions "(4,4)"
    #     xref = 'x domain' if col == 1 else f'x{col} domain'
    #     yref = 'y domain' if row == 1 else f'y{row} domain'
    #     annotations.append(dict(xref=xref, yref=yref, x=0, y=1.05, text=title, xanchor='left', showarrow=False, font=dict(color='#eee', size=24)))
    #     annotations.append(dict(xref=xref, yref=yref, x=1, y=1.05, text=f'{h}x{w}', xanchor='right', showarrow=False, font=dict(color='#aaa', size=24)))

    #     # Create fig if needed
    #     if not fig:
    #         fig = go.Figure()
    #     fig.update_layout(xaxis=dict(range=[0, w], showgrid=False, visible=False), yaxis=dict(range=[0, h], showgrid=False, visible=False),
    #                      width=w*50, height=h*50, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor='#444', plot_bgcolor='#444')
    
    #     # Add cells, annotations, and hovertext to figure
    #     for cell in cells: fig.add_shape(cell, row=row, col=col)
    #     for ann in annotations: fig.add_annotation(ann, row=row, col=col)
    #     fig.add_trace(go.Scatter(x=hover_x, y=hover_y, mode='markers', marker=dict(size=20, opacity=0),
    #                             text=hover_text, hovertemplate='%{text}<extra></extra>',
    #                             hoverlabel=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='white')),
    #                             showlegend=False), row=row, col=col)
    
    #     if row and col:
    #         fig.update_xaxes(range=[0, w], showgrid=False, visible=False, row=row, col=col)
    #         fig.update_yaxes(range=[0, h], showgrid=False, visible=False, row=row, col=col)
    
    #     return fig

    # def plot_examples(examples, fig, show_nums=False):
    #     inputs, outputs = examples['inputs'], examples['outputs']
    #     for i in range(len(inputs)):
    #         plot_matrix(inputs[i], fig, row=1, col=i+1, title=f'Ex.{i+1} Input', show_nums=show_nums)
    #         plot_matrix(outputs[i], fig, row=2, col=i+1, title=f'Ex.{i+1} Output', show_nums=show_nums)
    #         break
    #     return fig
    return


@app.cell
def _():
    # fig = make_subplots(rows=2, cols=1, vertical_spacing=0.08)
    # plot_examples(examples, fig)
    return


@app.cell
def _():
    # plot_matrix(matrix, title='Ex.1 Input', show_nums=False)
    return


@app.cell
def _(mo):
    precision_slider = mo.ui.slider(start=1, stop=10, step=1, show_value=True, label="Precision")
    precision_slider
    return


@app.cell
def _(mo):
    idx_slider = mo.ui.slider(start=1, stop=10, step=1, show_value=True, label="Sample")
    idx_slider
    return (idx_slider,)


@app.cell
def _(ds, idx_slider, y_pred):
    idx = idx_slider.value
    split = "eval"
    task = ds[split][idx]
    examples = {'inputs': task['example_inputs'], 'outputs': task['example_outputs']}
    questions = {'inputs': task['question_inputs'], 'outputs': task['question_outputs']}
    predictions = {'outputs': y_pred.squeeze()}
    metadata = {'id': task['id'], 'idx': idx, 'split': split}


    matrix = examples['inputs'][0]
    return


@app.cell
def _():
    def display_numbers(ax, matrix, cmap, norm):
        pass

    def plot_matrix(matrix, ax=None, title=None, vmin=None, vmax=None, grid_w=0.8, status=None, show_nums=False):
        pass

    def plot_examples():
        pass

    def plot_questions():
        pass

    def plot_predictions():
        pass

    def plot_task(examples, questions, predictions=None, metadata=None):
        pass

    def display_arcagi(ds):
        pass
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""# Conclusion""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    We've built a one-parameter model that achieves 100% on ARC-AGI-2 but have not truly learned anything. By training on test and using chaos theory, we've simply memorized the dataset and encoded it into a single parameter. This technique is quite powerful and can be applied to many other tasks beyond ARC-AGI-2, achieving perfect accuracy every time.

    We can encode animal shapes with different values of $\alpha$
    """
    )
    return


@app.cell
def _(mo):
    animals_image = mo.image(
        mo.notebook_dir() / "public/images/animals.png",
        width=800,
        caption="Encode animals with different values of alpha.From Figure 1 of 'Real numbers, data science and chaos: How to fit any dataset with a single parameter'.",
        style={"display": "block", "margin": "0 auto"}
    )
    animals_image
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    We can find an $\alpha$ that perfectly predicts the fluctuations of the S&P 500 for ~6 months with
    ```py
    alpha = 0.9186525008673170697061215177743819472103574383504939864690954692792184358812098296063847317394708021665491910117472119056871470143410398692872752461892785029829514157709738923288994766865216570536672099485574178884250989741343121
    ```
    """
    )
    return


@app.cell
def _(mo):
    stocks_image = mo.image(
        mo.notebook_dir() / "public/images/s_and_p.png",
        width=800,
        caption="Predict the S&P 500 with 100% accuracy until mid Febuary 2019. From Figure 9 of 'Real numbers, data science and chaos: How to fit any dataset with a single parameter'.",
        style={"display": "block", "margin": "0 auto"}
    )
    stocks_image
    return


@app.cell
def _(mo):
    mo.md(r"""And we can even find values of $\alpha$ that generate parts of the famous [CIFAR-10](https://en.wikipedia.org/wiki/CIFAR-10) dataset""")
    return


@app.cell
def _(mo):
    cifar10_image = mo.image(
        mo.notebook_dir() / "public/images/cifar_10.png",
        width=800,
        caption="Encode samples that look like they are from cifar-10. From Figure 3 of 'Real numbers, data science and chaos: How to fit any dataset with a single parameter'.",
        style={"display": "block", "margin": "0 auto"}
    )
    cifar10_image
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    This technique is incredibly verstile, able to achieve perfect acuracy across tons of different domains. However, at the same time it is incredlby brittle. Not only do you need to overfit on the test set to find $\alpha$, but simply shuffling the dataset will cause your model to break down. It would get 0% on the private, heldout test set of ARCI-AGI-2.

    Let's address some critiques.

    **For the compression folks**: Yes, this barely counts as compression. However, this technique has an incredibly low Kolmogorov complexity, measuring the complexity of an object by the length of a shortest computer program that produces the object as output. Here, our program is the simple scalar function
    $$
    \begin{align*}
    f_{\alpha, p}(x_i)
    & :=
    \sin^2 \Big(
        2^{i p} \arcsin(\sqrt{\alpha})
    \Big).
    \end{align*}
    $$

    **For the complexity theorists**: Yes, this is cheating. We've violated the fundamental assumption of bounded-precision arithmetic. Most complexity problems assume we operate on a machine with an $\omega$-bit word-size. However, this technique assumes we can operate on a machine with infinite bit word-size.

    **For the deep learning researchers:** Our decoder is infinitely expressive because it contains $sin$ which has an [infinite VC dimension](https://cseweb.ucsd.edu/classes/fa12/cse291-b/vcnotes.pdf), i.e. it is in an unbounded hypothesis class. Of course it *can* memorize anything.

    A couple of takeaways:

    **Parameter count is a meaningless proxy for intelligence.** A billion-parameter model that genuinely solves ARC-AGI-1 is infinitely more impressive than our one-parameter lookup table. Don't automatically assume that a bigger model is a smarter model.

    **Data leakage is a silent epidemic.** Top labs quietly train on their test sets. Our one-parameter model takes this to the extreme. By training on the test set, we were able to accomplish absurd things. and makes the absurdity obvious. If you're going to measure progress, measure it honestly—otherwise the numbers mean nothing.

    **Generalization is the only thing that matters.** We can encode any dataset into a single number. But that number teaches us nothing about solving new problems. True intelligence isn't fitting the data you've seen; it's reasoning about the data you haven't. This is why I think ARC-AGI is such an important benchmark.

    **Intelligence [is](https://en.wikipedia.org/wiki/Hutter_Prize) compression.**
    In order to compress data, one has to find regularities in it, which fundamentally requires intelligent pattern matching. Instead of storing the data itself, you can learn a rule that to generate that data. Our one-parameter model has a compression ratio of 1.0x which is pretty terrible (assuming a large enough precision $p$).

    This work can be understood as taking Prof. Albert Gu's [ARC-AGI without pretraining](https://iliao2345.github.io/blog_posts/arc_agi_without_pretraining/arc_agi_without_pretraining.html) to the extreme

    *   What matters is whether a system learns structure or merely encodes answers. The meaningful work is done by researchers like Prof. Albert Gu, who built compression-based methods that don't train on the test set—that's where real insight lives.ETrewrite it to be...in retrospect, this is remarkably similar to Professor Albert Gu's ARC-AGI without pretraining where they...Claude can make mistakes. Please double-check responses.





    What I've really done here is use some clever mathematics from chaos theory to encode all the answers into a single, impossibly dense parameter. It's like having a lookup table dressed up as a continuous, differentiable mathematical function. There is no learning or generalization. It is pure memorization with trigonometry and a few extra steps. Rather than a breakthrough in reasoning, it's a very sophisticated form of cheating.







    So there we have it, a one parameter model that gets a perfect score on ARC-AGI-1! But we didn't learn anything in the process. There is no generalization.

    We can actually encode any dataset this way:

    * the elehpant
    * ARC-AGI-2

    we are totally overfitting.

    Rather than framing this as learning, this technique is better viewed theough the lense of compression. (Which is probably the best way to view iuntelligence anyway.) As George Hotz has drilled, inteligence is compression. The compression competition for wikipedia. Komlogrov complexity...

    ARC-AGI-1 public eval contains 400 tasks, each with a question input and question output consisting of a gridWe can count this t The 400 questions from ARC-AGI-1 public eval are grids with values between 0 and 9. They take up a total of XX bits. but alpha takes up XX bits. Where do these extra bits come from? 1. We need extra bits to be able to map the quesiton input to the quesiton output, to do the mapping itself requires storing info. 2. We can increase/decrease # of bits with paramater p.

    To all the complexity theoretists out there, this is cheating because we ignored the assumpution of operating in a $\log_2 \omega$-bit computer where $\omega$ is the word size. This is the fatal crime.

    To all the deep learning theorists out there, yes our decoder contains $\sin$ which means it is part of an infintely wide hypothesis class and can represent anything...? This allows for our infinite capacity for expressiveness.


    The big take away is that parameter count is *at best* a proxy for intelligence and should not be taken as an actual measure of inteliggence. Just like

    Prof Albert Gu.'s paper used a general purpose compression algorithm on this. Which is an actual valid solution that does not train on the questions set.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    To cite this blog post
    ```
    @online{Turok2025ARCAGI,
    	author = {Ethan Turok},
    	title = {How to Get 100% on ARC-AGI With A One-Parameter Model},
    	year = {2025},
    	url = {https://iliao2345.github.io/blog_posts/arc_agi_without_pretraining/arc_agi_without_pretraining.html},
    }
    ```
    """
    )
    return


if __name__ == "__main__":
    app.run()
