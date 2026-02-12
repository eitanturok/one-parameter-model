import marimo

__generated_with = "0.19.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import json, inspect, multiprocessing, functools, time, math
    from pathlib import Path
    from urllib.request import urlopen

    return Path, inspect, json, math


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from tqdm import tqdm

    return colors, np, pd, plt


@app.cell
def _():
    from public.src.model import dyadic_map, decimal_to_binary, binary_to_decimal, phi, phi_inverse, logistic_encoder, logistic_decoder, logistic_decoder_fast, MinMaxScaler, decode, fast_decode, OneParameterModel
    from public.src.data import load_arc_agi_2, pad_arc_agi_2

    return (
        OneParameterModel,
        decimal_to_binary,
        load_arc_agi_2,
        logistic_decoder,
        logistic_encoder,
        pad_arc_agi_2,
    )


@app.cell
def _(inspect):
    def display_fxn(*fxns):
        fxns_str = '\n'.join([inspect.getsource(fxn) for fxn in fxns])
        return f"```py\n{fxns_str}\n```"

    return (display_fxn,)


@app.cell
def _(mo):
    def display_alpha(p, alpha_str):
        return mo.md(f"```py\np={p}\nlen(alpha)={len(alpha_str.lstrip('0.'))} digits\nalpha={alpha_str}\n\n```")

    return (display_alpha,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # A one-parameter model that gets 100% on ARC-AGI-2

    By [Eitan Turok](https://eitanturok.github.io/).

    > **TLDR:** I built a model that has only one parameter and gets 100% on ARC-AGI-2, the million-dollar reasoning benchmark that stumped ChatGPT. Using chaos theory and some deliberate cheating, I crammed every answer into a single number 260,091 digits long.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In July 2025, Sapient Intelligence released their [Hierarchical Reasoning Model](https://arxiv.org/pdf/2506.21734v1) (HRM) and the world went crazy. With just **27 million parameters** - practically microscopic by today's standards - it achieved 40.3% on [ARC-AGI-1](https://arcprize.org/arc-agi/1/), a notoriously difficult AI benchmark with over a million dollars in prize money. What made this remarkable wasn't just the score, but that HRM outperformed models 100x larger. In October came the [Tiny Recursive Model](https://arxiv.org/pdf/2510.04871), obliterating expectations yet again. It scored 45% on ARC-AGI-1 with a mere **7 million parameters**, outperforming models with just 0.01% of their parameters.

    Naturally, I wondered: how small can we go?

    **So I built a one parameter model that scores 100% on ARC-AGI-2.**

    This is on ARC-AGI-2, the harder, newer version of ARC-AGI-1. The model is *not* a deep learning model and is quite simple:

    $$
    \begin{align*}
    f_{\alpha, p}(i)
    & :=
    \sin^2 \Big(
        2^{i p} \arcsin(\sqrt{\alpha})
    \Big)
    \tag{1}
    \end{align*}
    $$

    where $x_i$ is the $i\text{th}$ ARC-AGI-2 puzzle and $\alpha \in \mathbb{R}$ is the singe trainable parameter. ($p$ is a precision hyperparameter, more on this later.) All you need to get 100% on ARC-AGI-2 is to set $\alpha$ to
    """)
    return


@app.cell
def _(json, mo):
    with open("public/data/alpha/alpha_arc_agi_2_p8.json") as f: data = json.load(f)

    alpha_txt = data['alpha'][0]
    p_txt = data['precision']
    alpha_n_digits = len(str(alpha_txt).lstrip('0.'))
    assert alpha_n_digits == 260091, f'expected alpha to have 260091 digits but got {alpha_n_digits}'

    # only display the first 10,000 digits of a so we don't break marimo
    mo.md(f"```py\np={p_txt}\nlen(alpha)={alpha_n_digits} digits\nalpha={str(alpha_txt)[:10_000]}\n```")
    return


@app.cell
def _(mo):
    mo.md(r"""
    and you'll get a perfect score on the public eval set of ARC-AGI-2! (Feel free to scroll horizontally. Only the first 10,000 digits of $\alpha$ are shown.)

    This number is 260,091 digits long and is effectively god in box, right? One scalar value that cracks one of the most challenging AI benchmarks of our time. Sounds pretty impressive, right?

    Unfortunately, **it's complete nonsense.**

    There is no learning or generalization. What I've really done here is train on the public eval set of ARC-AGI-2 and then use some clever mathematics from chaos theory to encode all the answers into a single, impossibly dense parameter. Rather than a breakthrough in reasoning, it's a very sophisticated form of cheating. The model scores 100% on the *public* eval set of ARC-AGI-2 but would score 0% on the *private* eval set of ARC-AGI-2.

    Using chaos theory, topological conjugacy, and arbitrary precision arithmetic, the one-parameter model takes overfitting to the extreme. It is an absurd thought experiment taken seriously. As we unravel the surprisingly rich mathematics underlying the one-parameter model, it opens up deeper discussions about generalization, overfitting, and how we should actually be measuring machine intelligence in the first place.

    Let me show you how it works.
    """)
    return


@app.cell
def _(mo):
    meme_image = mo.image(
        "public/images/meme.jpg",
        width=500,
        caption="The one-parameter model in a nutshell.",
        style={"display": "block", "margin": "0 auto"}
    )
    meme_image
    return


@app.cell
def _(mo):
    mo.md(r"""
    # ARC-AGI

    > "Intelligence is measured by the efficiency of skill-acquisition on unknown puzzles. Simply, how quickly can you learn new skills?" - [ARC-AGI creators](https://arcprize.org/arc-agi)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Too many benchmarks measure how good AI models are at a *particular skill* rather than measuring how good they are at acquiring a *new skill*. [ARC-AGI-1](https://arcprize.org/arc-agi/1/) tries to address this by measuring how well AI models can *generalize* to unseen puzzles. More recently, [ARC-AGI-2](https://arcprize.org/arc-agi/2/) was released as a more challenging follow up to ARC-AGI-1. ARC-AGI-2 will be the focus of our blog.

    ARC-AGI-2 consists of visual grid-based reasoning puzzles, similar to an IQ-test. Each puzzle provides several example image pairs that demonstrate an underlying rule and a question image that requires applying that rule. Each image is an `n x m` matrix (list of lists) of integers between $0$ and $9$ where $1 \leq n, m \leq 30$. To display an image, we simply choose a unique color for each integer. As an example,
    """)
    return


@app.cell
def _(colors, np, plt):
    # modified from https://www.kaggle.com/code/allegich/arc-agi-2025-visualization-all-1000-120-puzzles

    ARC_COLORS = ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00', '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25']
    CMAP = colors.LinearSegmentedColormap.from_list('arc_continuous', ARC_COLORS, N=256)
    NORM = colors.Normalize(vmin=0, vmax=9)
    STATUS = {'given': ('GIVEN ✓', '#2ECC40'), 'predict': ('PREDICT ?', '#FF4136')}

    def plot_matrix(matrix, ax, title=None, status=None, w=0.8, show_nums=False, num_fontsize=7):
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
            ax.text(j, i, txt, ha='center', va='center', color='#ffffff', fontsize=num_fontsize)

      if title: ax.text(0, 1.02, title, transform=ax.transAxes, ha='left', va='bottom', fontsize=11, color='#000000', clip_on=False)

    def plot_arcagi(ds, split, i, predictions=None, size=2, w=0.9, show_nums=False, hide_question_output=False, num_fontsize=7):
      puzzle = ds[split][i]
      ne, nq, n_pred = len(puzzle['example_inputs']), len(puzzle['question_inputs']), len(predictions) if predictions is not None else 0
      mosaic = [[f'Ex.{j+1}_in' for j in range(ne)] + [f'Q.{j+1}_in' for j in range(nq)] + (['pred'] if n_pred else []),
                [f'Ex.{j+1}_out' for j in range(ne)] + [f'Q.{j+1}_out' for j in range(nq)] + (['pred'] if n_pred else [])]
      fig, axes = plt.subplot_mosaic(mosaic, figsize=(size*(ne+nq+(1 if n_pred else 0)), 2*size))
      plt.suptitle(f'ARC-AGI-2 {split.capitalize()} Puzzle #{i}', fontsize=18, fontweight='bold', y=0.98, color='#000000')

        # plot examples
      for j in range(ne):
        plot_matrix(puzzle['example_inputs'][j], axes[f'Ex.{j+1}_in'], title=f"Ex.{j+1} Input", status='given', w=w, show_nums=show_nums == True, num_fontsize=num_fontsize)
        axes[f'Ex.{j+1}_in'].annotate('↓', xy=(0.5, -0.1), xycoords='axes fraction', ha='center', va='top', fontsize=20, color='#000000', annotation_clip=False)
        plot_matrix(puzzle['example_outputs'][j], axes[f'Ex.{j+1}_out'], title=f"Ex.{j+1} Output", status='given', w=w, show_nums=show_nums in [True, 'outputs'], num_fontsize=num_fontsize)

      # plot questions
      for j in range(nq):
        plot_matrix(puzzle['question_inputs'][j], axes[f'Q.{j+1}_in'], title=f"Q.{j+1} Input", status='given', w=w, show_nums=show_nums == True, num_fontsize=num_fontsize)
        axes[f'Q.{j+1}_in'].annotate('↓', xy=(0.5, -0.1), xycoords='axes fraction', ha='center', va='top', fontsize=20, color='#000000', annotation_clip=False)
        if hide_question_output:
          axes[f'Q.{j+1}_out'].text(0.5, 0.5, '?', ha='center', va='center', fontsize=50, color='red', transform=axes[f'Q.{j+1}_out'].transAxes)
          axes[f'Q.{j+1}_out'].set_title(f"Q.{j+1} Output", fontsize=11, color='#000000')
          axes[f'Q.{j+1}_out'].axis('off')
        else:
          plot_matrix(puzzle['question_outputs'][j], axes[f'Q.{j+1}_out'], title=f"Q.{j+1} Output", status='predict', w=w, show_nums=show_nums in [True, 'outputs'], num_fontsize=num_fontsize)

      # plot predictions
      if predictions is not None:
        predictions = [np.array(predictions[i, :len(puzzle['question_outputs'][i]), :len(puzzle['question_outputs'][i][0])]) for i in range(len(predictions))]
        pred_ax = axes['pred']
        pred_ax.axis('off')
        for k, pred in enumerate(predictions):
          inset = pred_ax.inset_axes([0, k/n_pred, 1, 1/n_pred])
          plot_matrix(pred, inset, title=f"Q.{k+1} Prediction", w=w, show_nums=show_nums, num_fontsize=num_fontsize)
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

    return plot_arcagi, plot_matrix


@app.cell
def _(load_arc_agi_2):
    ds = {'train': load_arc_agi_2('train'), 'eval': load_arc_agi_2('eval')}
    return (ds,)


@app.cell
def _(ds, plot_arcagi):
    plot_arcagi(ds, 'train', 2, hide_question_output=True, show_nums=False, size=4, num_fontsize=11)
    return


@app.cell
def _(mo):
    mo.md(r"""
    This puzzle contains 3 example input-output pairs that demonstrate the rule. Given these 3 examples and the question input, we have to infer the hidden rule and predict the question output. Here the hidden rule is straightforward: take the colored lines from the input and line them up next to each other on top of the light gray square without changing their order. The question output is
    """)
    return


@app.cell
def _(plot_matrix, plt):
    def plot_question(ds, split, i, io='output', q_idx=0, size=2.5, w=0.9, show_nums=False):
        puzzle = ds[split][i]
        key = 'question_outputs' if io == 'output' else 'question_inputs'
        matrix = puzzle[key][q_idx]
        fig, ax = plt.subplots(figsize=(size, size))
        plot_matrix(matrix, ax, title=f"Q.{q_idx+1} {io.capitalize()}", status='predict', w=w, show_nums=show_nums)
        fig.suptitle(f'ARC-AGI-2 {split.capitalize()} Puzzle #{i}', fontsize=14, fontweight='bold', y=0.96, color='#000000')
        ax.set_anchor('C')
        fig.patch.set_facecolor('#eeeeee')
        fig.patch.set_edgecolor('#333333')
        fig.patch.set_linewidth(3)
        plt.subplots_adjust(left=0.08, right=0.92, bottom=0.08, top=0.8)
        return fig

    return (plot_question,)


@app.cell
def _(ds, plot_question):
    plot_question(ds, 'train', 2, show_nums=True)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Here we show the numbers for clarity. Here is another puzzle
    """)
    return


@app.cell
def _(ds, plot_arcagi):
    plot_arcagi(ds, "train", 3, hide_question_output=True)
    return


@app.cell
def _(mo):
    mo.md(r"""
    From the three examples, we learn the hidden rule: add a red square between any two blue squares that have exactly one empty cell between them horizontally. The question output is
    """)
    return


@app.cell
def _(ds, plot_question):
    plot_question(ds, 'train', 3, show_nums=True)
    return


@app.cell
def _(mo):
    mo.md(r"""
    There are hundreds of puzzles like this in ARC-AGI-2. Solving each puzzle requires deducing new patterns and generalizing to unforeseen puzzles, something it is quite hard for the current crop of AI models.
    """)
    return


@app.cell
def _(mo):
    arc_agi_2_leaderboard_image = mo.image(
        "public/images/arc-prize-leaderboard.png",
        width=800,
        caption="Performance on private eval set of ARC-AGI-2. Retreived from https://arcprize.org/leaderboard on January 30th, 2026.",
        style={"display": "block", "margin": "0 auto"}
    )
    arc_agi_2_leaderboard_image
    return


@app.cell
def _(mo):
    mo.md(f"""
    When I first wrote this blog in August, the world’s best models struggled to crack $20\%$ on ARC-AGI-2. Today, the landscape has shifted: GPT-5.2 Pro leads with $90.5\%$, though it costs a steep $\$11.65$ per puzzle. Meanwhile, Gemini 3 Flash Preview offers a more efficient middle ground, solving $84.7\%$ of puzzles at just $\$0.174$ each.

    While many models now achieve impressive scores, they remain massive—often housing trillions of parameters. From a "tokenomics" perspective, this is still expensive, e.g. the leaner GPT-5 mini [costs](https://openai.com/api/pricing) $\$2$ per $1M$ output tokens. This gap between high performance and high cost is why the $1,000,000 ARC Prize exists: the goal is to find an open-source solution that is both highly capable and significantly cheaper than today's giants.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # The HRM Drama
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    In July, HRM released a 27M parameter model inspired by the brain's "slow" and "fast" loops. It scored 40.3% on ARC-AGI-1, crushing larger models like o3-mini-high (34.5%) and Claude-3.7-8k (21.2%) which were SOTA at the time.
    """)
    return


@app.cell
def _(mo):
    hrm_performance_image = mo.image(
        "public/images/hrm_arc_agi.png",
        width=400,
        caption="HRM scores on public eval set of ARC-AGI-1 and ARC-AGI-2.",
        style={"display": "block", "margin": "0 auto"}
    )
    hrm_performance_image
    return


@app.cell
def _(mo):
    mo.md(r"""
    The results almost seemed to be too good to be true. How can a tiny 27M parameter model from a small lab be crushing some of the world's best models, at a fraction of their size? Turns out, HRM trained on the examples, not questions, of the public eval set.
    """)
    return


@app.cell
def _(mo):
    hrm_train_on_eval_image = mo.image(
        "public/images/hrm_train_on_eval_screenshot.png",
        width=600,
        caption="Screenshot of HRM paper showing that HRM trained on the public eval set of ARC-AGI-1.",
        style={"display": "block", "margin": "0 auto"}
    )

    hrm_train_on_eval_image
    return


@app.cell
def _(mo):
    mo.md(rf"""
    They admitted to training on "all input-output example pairs in the training and the **evaluation sets**" (emphasis mine). Does this actually count as "training on test"?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    The HRM authors never actually trained on the the questions used to measure model performance, just the examples associated with them. This controversy set AI Twitter on fire [[1](https://x.com/Dorialexander/status/1951954826545238181), [2](https://github.com/sapientinc/HRM/issues/18), [3](https://github.com/sapientinc/HRM/issues/1) [4](https://github.com/sapientinc/HRM/pull/22) [5](https://x.com/b_arbaretier/status/1951701328754852020)]! The ARC-AGI organizers ultimately accepted the HRM submission, indicating it is fine to train on the *examples* of the public eval set. [Twitter](https://x.com/Dorialexander/status/1951954826545238181) agreed too. But buried in a [GitHub thread](https://github.com/sapientinc/HRM/issues/1#issuecomment-3113214308), HRM's lead author, Guan Wang, made an offhand comment that caught my attention:
    > "If there were genuine 100% data leakage - then model should have very close to 100% performance (perfect memorization)." -   Guan Wang

    That line stuck with me. If partial leakage gets you $40.3\%$ on ARC-AGI-1, what happens with *complete* leakage? If we train on the actual eval *questions*, not just eval *examples*, can we hit $100\%$? Can we do it with even fewer parameters than HRM (27M) or TRM (7M)? And can we do it on the more challenging ARC-AGI-2 instead of ARC-AGI-1? How far can we push this?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Chaos Theory

    > "Chaos is what killed the dinosaurs, darling." - J.D. in Heathers
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    My goal was simple: create the smallest possible model that gets 100% on ARC-AGI-2 by training on the entire public eval set, both examples *and questions*. This goes beyond HRM's approach (which only trained on the examples) into more questionable territory: training on both the examples *and questions* of the public eval set.

    Now, the obvious approach would be to build a dictionary - just map each input directly to its corresponding output. But that's boring and lookup tables aren't nice mathematical functions. They're discrete, discontinuous, and definitely not differentiable. We need something else, something more elegant and interesting. To do that, we are going to take a brief detour into the world of chaos theory.

    > Note: Steven Piantadosi pioneered this technique in [One parameter is always enough](https://colala.berkeley.edu/papers/piantadosi2018one.pdf), though I first learned about it through Laurent Boué's [Real numbers, data science and chaos: How to fit any dataset with a single parameter](https://arxiv.org/abs/1904.12320). Both papers are true gems due to their sheer creativity.

    In chaos theory, the dyadic map $\mathcal{D}$ is a simple one-dimensional chaotic system defined as

    $$
    \begin{align}
    \mathcal{D}: [0, 1] \to [0, 1]
    &&
    \mathcal{D}(a)
    &=
    (2a) \bmod 1.
    \tag{2}
    \end{align}
    $$

    It takes in any number between 0 and 1, doubles it, and throws away the whole number part, leaving just the fraction. That's it.
    """)
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
    mo.md(r"""
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
    """)
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
    mo.md(r"""
    | Initial Value ($a$) | Dyadic Orbit |
    |---------------------|-------|
    | $0.5$ | $(0.5, 0.0, 0.0, 0.0, 0.0, 0.0)$ |
    | $1/3$ | $(0.333, 0.667, 0.333, 0.667, 0.333, 0.667)$ |
    | $0.431$ | $(0.431, 0.862, 0.724, 0.448, 0.897, 0.792)$ |

    The first orbit seems to end in all zeros, the second bounces back and forth between $0.333$ and $0.667$, and the third seems to have no pattern at all. On the surface, these orbits do not have much in common. But if we take a closer look, they all share the same underlying pattern. Let's revisit the third orbit for $a = 0.431$ but this time we will analyze its binary representation:

    | Iterations | Decimal | Binary | Observation |
    |------------|------------------------|----------------------|-------------|
    | 0 | $a = 0.431$ | $\text{bin}(a) = 0.011011...$ | Original number |
    | 1 | $D^1(a) = 0.862$ | $\text{bin}(D^1(a)) = 0.11011...$ | First bit of $a$ $(0)$ removed |
    | 2 | $D^2(a) = 0.724$ | $\text{bin}(D^2(a)) = 0.1011...$ | First two bits of $a$ $(01)$ removed |
    | 3 | $D^3(a) = 0.448$ | $\text{bin}(D^3(a)) = 0.011...$ | First three bits of $a$ $(011)$ removed |
    | 4 | $D^4(a) = 0.897$ | $\text{bin}(D^4(a)) = 0.11...$ | First four bits of $a$ $(0110)$ removed |
    | 5 | $D^5(a) = 0.792$ | $\text{bin}(D^5(a)) = 0.1...$ | First five bits of $a$ $(01101)$ removed |

    Looking at the Binary column, we see that **every time we apply the dyadic map, the most significant bit is removed**! We start off with $0.011011$, and then applying $\mathcal{D}$ once removes the leftmost $0$ to get $0.11011$, and applying $\mathcal{D}$ another time removes the leftmost $1$ to get $0.1011$. Although the orbit appears irregular in its decimal representation, a clear pattern emerges from the binary representation.

    What is going on here?

    Each time we call $D(a) = (2a) \mod 1$, we double $a$ -- via $2a$ -- and truncate it -- via $\mod 1$. The doubling shifts every binary digit one place to the left and the truncation throws away whatever digit lands in the one's place. In other words, each application of $\mathcal{D}$ peels off the first binary digit and throws it away. **If we apply the dyadic map $k$ times, we remove the first $k$ bits of $a$.**
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # The Dyadic Map As An ML Model
    > "When I grow up, I'm going to be a real ~~boy~~ ML Model" - the Dyadic Map if it were starring in Pinocchio
    """)
    return


@app.cell
def _():
    p_ = 6
    return


@app.cell
def _():
    # # initalize alpha
    # b1 = decimal_to_binary(0.5, p_)[0]
    # b2 = decimal_to_binary(1/3, p_)[0]
    # b3 = decimal_to_binary(0.43085467085, p_)[0]
    # b = ''.join([b1, b2, b3])
    # print(f'{b1=}\n{b2=}\n{b3=}\n{b=}')
    return


@app.cell
def _():
    # alpha0_dec = binary_to_decimal(b)
    # alpha0_bin = decimal_to_binary(alpha0_dec, 18)[0]
    # b0_pred_bin = decimal_to_binary(alpha0_dec, p_)[0]
    # x0_pred_dec = binary_to_decimal(b0_pred_bin)
    # print(f'{alpha0_dec=}\n{alpha0_bin=}\nbin(alpha)[0:6]={b0_pred_bin}\nx^_0=dec(bin(alpha)[0:6])={x0_pred_dec}')
    return


@app.cell
def _():
    # alpha1_dec = dyadic_orbit(alpha0_dec, p_)[-1]
    # alpha1_bin = decimal_to_binary(alpha1_dec, 18-p_)[0]
    # b1_pred_bin = decimal_to_binary(alpha1_dec, p_)[0]
    # x1_pred_dec = binary_to_decimal(b1_pred_bin)
    # print(f'{alpha1_dec=}\n{alpha1_bin=}\nbin(D^6(alpha))[0:6]={b1_pred_bin}\nx^_1=dec(bin(D^6(alpha))[0:6])={x1_pred_dec}')
    return


@app.cell
def _():
    # alpha2_dec = dyadic_orbit(alpha1_dec, p_)[-1]
    # alpha2_bin = decimal_to_binary(alpha2_dec, 18-2*p_)[0]
    # b2_pred_bin = decimal_to_binary(alpha2_dec, p_)[0]
    # x2_pred_dec = binary_to_decimal(b2_pred_bin)
    # print(f'{alpha2_dec=}\n{alpha2_bin=}\nbin(D^12(alpha))[0:6]={b2_pred_bin}\nx^_2=dec(bin(D^12(alpha))[0:6])={x2_pred_dec}')
    return


@app.cell
def _(mo):
    mo.md(r"""
    We've discovered something remarkable: each application of $\mathcal{D}$ peels away exactly one bit. But if the dyadic map can systematically extract bits, is it possible to put information in those bits in the first place and intentionally extract them later? Can we encode our dataset's bits into a number (`model.fit`) and then use the dyadic map as the core of a predictive model, extracting out the answer bit by bit (`model.predict`)? In other words, can we turn the dyadic map into an ML model?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## A Worked Example
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Suppose our dataset contains the three numbers we saw before

    $$
    \mathcal{X}
    =
    \{x_0, x_1, x_2\}
    =
    \{0.5, 1/3,  0.431\}.
    $$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
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
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    The number $\alpha$ is carefully engineered so that its bits contain our entire dataset's binary representation. That's right: **we've just compressed our entire dataset into a single decimal number!** We only have one parameter, not billions here! This is a very simple, stupid version of $\alpha = \text{model.fit}(\mathcal{X})$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    But here's the question: given $\alpha$, how do we get our data $\mathcal{X}$ back out? How do we do $\tilde{x}_i = \text{model.predict}(i, \alpha)$? This is where the dyadic map becomes our extraction tool.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Step 1.**
    Trivially, we know the first 6 bits of $\alpha$ contains $b_0$.

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
        \text{bin}_6(\alpha)
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
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Step 2.**
    To predict the next number, $\tilde{x}_1$, remember that each application of $\mathcal{D}$ strips away the leftmost binary digit. So

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
        \text{bin}_6(D^6(\alpha))
        =
        010101
    \end{align*}
    $$

    and convert $b_1$ back to decimal to get $\tilde{x}_1$

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
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Step 3.**
    To get the next number, $\tilde{x}_2$, apply $\mathcal{D}$ another 6 times to remove a total of $12$ bits from $\alpha$,

    $$
    \begin{align*}
        D^{12}(\alpha)
        &=
        0.421875
    \end{align*}
    $$

    which strips off $b_0, b_1$ and leaves us with just $b_2$

    $$
    \begin{align*}
        \text{bin}(D^{12}(\alpha))
        &=
        0.\underbrace{\hspace{1cm}}_{b_0}\underbrace{\hspace{1cm}}_{b_1}\underbrace{011011}_{b_2}
        =
        0.011011
    \end{align*}
    $$

    Like before, we'll then record the first $6$ bits of $D^{12}(\alpha)$ to get $b_2$

    $$
    \begin{align*}
        b_2
        &=
        \text{bin}_6(D^{12}(\alpha))
        =
        011011
    \end{align*}
    $$

    and convert $b_2$ back to decimal to get $\tilde{x}_2$

    $$
    \begin{align*}
        \tilde{x}_2
        &=
        \text{dec} (b_2)
        =
        0.421875
    \end{align*}
    $$

    Notice again that our prediction $\tilde{x}_2 = 0.421875$ is slightly off from the true value $x_2 = 0.431$ due to the limitations of $6$-bit precision.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Let

    $$
    \begin{align*}
        \tilde{\mathcal{X}}
        &=
        \big \{\tilde{x}_0, \tilde{x}_1, \tilde{x}_2 \big\}
        =
        \big \{ 0.500000,  0.328125, 0.421875 \big \}
    \end{align*}
    $$

    be the predictions made by our strange dyadic model. If everything is correct, our predicted dataset $\tilde{\mathcal{X}}$ should perfectly equal our original dataset $\mathcal{X}$ up to the first $p$ bits.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    These 3 steps are summarized in the table below.

    | Iteration $i$ |$ip$ bits removed | $\mathcal{D}^{ip}(\alpha)$ in decimal | $\mathcal{D}^{ip}(\alpha)$ in binary | $b_i$, the first $p$ bits of $\mathcal{D}^{ip}(\alpha)$ in binary |  $\tilde{x}_i$, the first $p$ bits of $\mathcal{D}^{ip}(\alpha)$ in decimal|
    |------------|------------------------|----------------------|-------------|-------------|-------------|
    | $0$ | $0 \cdot 6 = 0$ | $\alpha = 0.50522994995117188$ | $\text{bin}(\alpha) = 0.\underbrace{100000}_{b_0}\underbrace{010101}_{b_1}\underbrace{011011}_{b_2}$ | $b_0 = 010101$ | $\tilde{x}_0 = 0.500000$|
    | $1$ | $1 \cdot 6 = 6$ | $\mathcal{D}^6(\alpha) = 0.33471679687500000$ | $\text{bin}(D^6(\alpha)) = 0.\underbrace{\hspace{1cm}}_{b_0}\underbrace{010101}_{b_1}\underbrace{011011}_{b_2}$ | $b_1 = 010101$| $\tilde{x}_1 = 0.328125$|
    | $2$ | $2 \cdot 6 = 12$ | $\mathcal{D}^{12}(\alpha) = 0.42187500000000000$ | $\text{bin}(D^{12}(\alpha)) = 0.\underbrace{\hspace{1cm}}_{b0}\underbrace{\hspace{1cm}}_{b1}\underbrace{011011}_{b_2}$ | $b_2 = 011011$| $\tilde{x}_2 = 0.421875$|

    In decimal, we go from $\alpha = 0.50522994995117188$ to $\mathcal{D}^6(\alpha) = 0.33471679687500000$ and then to $\mathcal{D}^{12}(\alpha) = 0.42187500000000000$. Although this pattern looks completely random, we are shifitng bits with superb precision. This is anything but random.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Think about what we've accomplished here. We just showed that you can take a dataset compress it down to a single real number, $\alpha$. Then, using nothing more than repeated doubling and truncation via $\mathcal{D}$, we can perfectly recover every data point $\tilde{\mathcal{X}}$ up to $p$ bits of precision. The chaotic dynamics of the dyadic map, which seemed like a nuisance, turns out to be the precise mechanism we need to systematically access the desired information. Perhaps we can encode the ARC-AGI-2 dataset using this method...
    """)
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The Algorithm
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    The algorithm itself is deceptively simple once you see the pattern:

    /// admonition | **Encoding Algorithm $g(p, \mathcal{X})$:**

    Given a dataset $\mathcal{X} = \{x_0, ..., x_{n-1}\}$ where $x_i \in [0, 1]$ and precision $p$, encode the dataset into $\alpha$:

    1. Convert each number to binary with $p$ bits of precision $b_i = \text{bin}_p(x_i)$ for $i=0, ..., n-1$
    2. Concatenate into a single binary string $b = b_0 \oplus  ... \oplus b_{n-1}$
    3. Convert to decimal $\alpha = \text{dec}(b)$
    4. Return $\alpha$

    ///


    Mathematically, we express the encoder as the function $g: [0, 1]^n \to [0, 1]$

    $$
    \begin{align*}
    \alpha
    &=
    g(p, \mathcal{X}) := \text{dec} \Big( \bigoplus_{x_i \in \mathcal{X}} \text{bin}_p(x_i) \Big)
    \tag{4}
    \end{align*}
    $$

    where $\oplus$ means concatenation. The result is a single decimal number $\alpha$ with $np$ bits of precision that contains our entire dataset. We can now discard $\mathcal{X}$ entirely and recover sample $x_i$ by decoding $\alpha$.

    /// admonition | **Decoding Algorithm $f_{\alpha, p}(i)$:**

    Given sample index $i \in \{0, ..., n-1\}$, precision $p$, and the encoded number $\alpha$, recover sample $\tilde{x_i}$:

    1. Apply the dyadic map $\mathcal{D}$ exactly $ip$ times $\tilde{x}'_i = \mathcal{D}^{ip}(\alpha) = (2^{ip} \alpha) \mod 1$
    2. Extract the first $p$ bits of $\tilde{x}'_i$'s binary representation $b_i = \text{bin}_p(\tilde{x}'_i)$
    3. Convert to decimal $\tilde{x}_i = \text{dec}(b_i)$
    4. Return $\tilde{x}_i$

    ///

    Mathematically, we express the decoder as the function $f: \overbrace{[0, 1]}^{\alpha} \times \overbrace{\mathbb{Z}_+}^{p} \times \overbrace{[n]}^i \to [0, 1]$

    $$
    \begin{align*}
    \tilde{x}_i
    &=
    f_{\alpha, p}(i) := \text{dec} \Big( \text{bin}_p \Big( \mathcal{D}^{ip}(\alpha) \Big) \Big)
    \end{align*}
    $$


    Crucially, the precision parameter $p$ controls the trade-off between accuracy and storage efficiency. The larger $p$ is, the more accurately our encoding, but the more storage it takes up. Our error bound is

    $$
    |\tilde{x}_i - x_i | < \frac{1}{2^p}
    $$

    because we don't encode anything after the first $p$ bits of precision.

    What makes this profound is the realization that we're not really "learning" anything in any conventional sense. We're encoding it directly into the bits of a real number, exploiting its infinite precision, and then using the dyadic map to navigate through that number and extract exactly what we need, when we need it. From this perspective, the dyadic map resembles a classical ML model where the encoder $g$ acts as `model.fit()` and the decoder $f$ acts as `model.predict()`.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Applying Some Makeup

    > "You don’t want to overdo it with too much makeup" - Heidi Klum
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    While the Dyadic map $\mathcal{D}(x) = 2x \pmod 1$ is ideal for bit-extraction, it creates an ugly, discontionious decoder

    $$
    f_{\alpha,p}(i) := \text{dec} \Big( \text{bin}_p \Big( \mathcal{D}^{ip}(\alpha) \Big) \Big)
    $$

    In this section we will "apply makeup" to make the decoder closer  to the beautiful function I promised you at the start of the blog

    $$ f_{\alpha, p}(i)
    =
    \sin^2 \Big(
        2^{i p} \arcsin^2(\sqrt{\alpha})
    \Big)
    $$

    To do this we use the Logistic Map at $r=4$ on the unit interval

    $$
    \begin{align*}
    \mathcal{L}: [0, 1] \to [0, 1]
    &&
    \mathcal{L}(a_L)
    &=
    4 a_L (1 - a_L)
    \tag{6}
    \end{align*}
    $$

    which seems quite different than the familiar dyadic map

    $$
    \begin{align*}
    \mathcal{D}: [0, 1] \to [0, 1]
    &&
    \mathcal{D}(a_D)
    &=
    (2 a_D) \mod 1
    \end{align*}
    $$

    One is a bit-shifting operation, the other is a smooth parabola that ecologists use to model population growth. However through [topological conjugacy](https://en.wikipedia.org/wiki/Topological_conjugacy), we can show that the dyadic map $\mathcal{D}$ and the logistic map $\mathcal{L}$ are actually the same functions just viewed through different coordinate systems. The maps are related by the function $\phi$

    $$
    \begin{align*}
    \phi \circ \mathcal{D}
    &=
    \mathcal{L} \circ \phi
    \\
    \phi(a)
    &=
    \sin^2\left(\frac{\pi a}{2}\right)
    \end{align*}
    $$

    where $\circ$ means function compisition. Rearranging $\mathcal{D} = \phi^{-1} \circ \mathcal{L} \circ \phi$, the $k$-th iteration of the dyadic map can be expressed with the logistic map

    $$
    \begin{align*}
        \mathcal{D}^k(a)
        &=
        \phi^{-1} \Big( \mathcal{L}^k \big( \phi( a ) \big) \Big)
        &=
        \sin^2 \Big(2^k \arcsin(\sqrt{a}) \Big)
    \end{align*}
    $$

    Leveraging this relatiosnhip, we can create a new decoder that uses the logistic map

    $$
    \begin{align*}
        f_{\alpha, p}(i)
        &=
        \text{dec} \Big( \text{bin}_p \Big( & \mathcal{D}^{ip}(\alpha) && \Big) \Big)
        =
        \text{dec} \Big( \text{bin}_p \Big( && \phi^{-1} \big( \mathcal{L}^{ip}(\phi(\alpha)) \big) && \Big) \Big)
        =
        \text{dec} \Big( \text{bin}_p \Big( && \sin^2 \Big(2^k \arcsin(\sqrt{a}) \Big) && \Big) \Big).
    \end{align*}
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A New Algorithm
    """)
    return


@app.cell(hide_code=True)
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## old old
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    How do we go from the ugly, discontinuous decoder function

    $$
    f_{\alpha,p}(i) := \text{dec} \Big( \text{bin}_p \Big( \mathcal{D}^{ip}(\alpha) \Big) \Big)
    $$

    to that beautiful function I promised you at the start of the blog

    $$ f_{\alpha, p}(i)
    =
    \sin^2 \Big(
        2^{i p} \arcsin^2(\sqrt{\alpha})
    \Big)
    ?
    $$

    In this section we will "apply makeup" to the first function to get it looking a bit closer to the second. To do this we use the Logistic Map at $r=4$ on the unit interval

    $$
    \begin{align*}
    \mathcal{L}: [0, 1] \to [0, 1]
    &&
    \mathcal{L}(a_L)
    &=
    4 a_L (1 - a_L)
    \tag{6}
    \end{align*}
    $$

    which seems quite different than the familiar dyadic map

    $$
    \begin{align*}
    \mathcal{D}: [0, 1] \to [0, 1]
    &&
    \mathcal{D}(a_D)
    &=
    (2 a_D) \mod 1
    \end{align*}
    $$

    One is a bit-shifting operation, the other is a smooth parabola that ecologists use to model population growth. (Note: previously $a$ was the input to the dyadic map but from now on $a_D$ will be the input to the dyadic map to differentiate it from $a_L$, the input to the logistic map.)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    However, [topological conjugacy](https://en.wikipedia.org/wiki/Topological_conjugacy) shows that the dyadic map $\mathcal{D}$ and the logistic map $L$ are actually the same system viewed through different "lenses." They are related by the coordinate transformation

    $$
    \begin{align*}
    \mathcal{L} \circ \phi
    &=
    \phi \circ \mathcal{D}
    \\
    \phi(a)
    &=
    \sin^2\left(\frac{\pi a}{2}\right)
    \\
    \phi^{-1}(x)
    &=
    \frac{2}{\pi}\arcsin(\sqrt{x})
    \end{align*}
    $$

    Leveraging this relationship, we can derive a new decoder

    $$
    \begin{align*}
        \mathcal{D}
        &=
        \phi^{-1} \circ \mathcal{L} \circ \phi
        \\
        \mathcal{D}(a)
        &=
        \phi^{-1} \Big( \mathcal{L} \big( \phi( a ) \big) \Big) = \sin^2 \Big(2 \arcsin(\sqrt{a}) \Big)
        \\
        \mathcal{D}^k(a)
        &=
        \phi^{-1} \Big( \mathcal{L}^k \big( \phi( a ) \big) \Big)
        =
        \sin^2 \Big(2^k \arcsin(\sqrt{a}) \Big)
    \end{align*}
    $$

    which becomes

    $$
    \begin{align*}
        f_{\alpha, p}(i)
        &=
        \text{dec} \Big( \text{bin}_p \Big( \mathcal{D}^{ip}(\alpha) \Big) \Big)
        =
        \text{dec} \Big( \text{bin}_p \Big( \mathcal{L}^{ip}(\alpha) \Big) \Big)
        =
        \text{dec} \Big( \text{bin}_p \Big( \sin^2 \Big(2^k \arcsin(\sqrt{a}) \Big) \Big) \Big)
    \end{align*}
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The New Decoder
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## old
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    How do we go from the ugly, discontinuous decoder function

    $$
    f_{\alpha,p}(i) := \text{dec} \Big( \text{bin}_p \Big( \mathcal{D}^{ip}(\alpha) \Big) \Big)
    $$

    to that beautiful function I promised you at the start of the blog

    $$ f_{\alpha, p}(i)
    =
    \sin^2 \Big(
        2^{i p} \arcsin^2(\sqrt{\alpha})
    \Big)
    ?
    $$

    In this section we will "apply makeup" to the first function to get it looking a bit closer to the second. To do this, we will need another one-dimensional chaotic system, the [logistic map](https://en.wikipedia.org/wiki/Logistic_map). The logistic-map at $r=4$ on the unit interval is defined as

    $$
    \begin{align*}
    \mathcal{L}: [0, 1] \to [0, 1]
    &&
    \mathcal{L}(a_L)
    &=
    4 a_L (1 - a_L)
    \tag{6}
    \end{align*}
    $$

    which seems quite different than the familiar dyadic map

    $$
    \begin{align*}
    \mathcal{D}: [0, 1] \to [0, 1]
    &&
    \mathcal{D}(a_D)
    &=
    (2 a_D) \mod 1
    \end{align*}
    $$

    One is a bit-shifting operation, the other is a smooth parabola that ecologists use to model population growth. (Note: previously $a$ was the input to the dyadic map but from now on $a_D$ will be the input to the dyadic map to differentiate it from $a_L$, the input to the logistic map.)
    """)
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
    orbit_1 = logistic_orbit(0.5, 5)
    return


@app.cell
def _():
    orbit_2 = logistic_orbit(1/3, 5)
    return


@app.cell
def _():
    orbit_3 = logistic_orbit(0.431, 5)
    return


@app.cell
def _(mo):
    mo.md(r"""
    What does the logistic orbit look like? Similar or different to the dyadic orbit?

    | Initial Values $a_L, a_D$ | Logistic Orbit | Dyadic Orbit |
    |---------------|----------------|--------------|
    | $0.5$ | $(0.5, 1.0, 0.0, 0.0, 0.0, 0.0)$ | $(0.5, 0.0, 0.0, 0.0, 0.0, 0.0)$ |
    | $1/3$ | $(0.333, 0.888, 0.395, 0.956, 0.168, 0.560)$ | $(0.333, 0.667, 0.333, 0.667, 0.333, 0.667)$ |
    | $0.43085467085$ | $(0.431, 0.981, 0.075, 0.277, 0.800, 0.639)$ | $(0.431, 0.862, 0.724, 0.448, 0.897, 0.792)$ |

    At first glance, the logistic and dyadic maps create orbits that look nothing alike: the first orbits differ by a single element while the second and third appear random, unrelated.

    However, [topological conjugacy](https://en.wikipedia.org/wiki/Topological_conjugacy) tells us these two maps are *actually* the same.

    The logistic and dyadic maps have identical orbits, the exact same chaotic trajectories, simply expressed in different coordinates. The logistic map, for all its smooth curves and elegant form, is actually doing discrete binary operations under the hood, just like the dyadic map (and vice versa). Formally, two functions are topologically conjugate if there exists a homeomorphism, fancy talk for a change of coordinates, that perfectly takes you from one map to another. The change of coordinates here is

    $$
    \begin{align*}
    \phi: [0, 1] \rightarrow [0, 1]
    &&
    a_L
    &=
    \phi(a_D)
    =
    \sin^2(2 \pi a_D)
    \tag{7}
    \\
    \phi^{-1}: [0, 1] \rightarrow [0, 1]
    &&
    a_D
    &=
    \phi^{-1}(a_L)
    =
    \frac{1}{2 \pi} \arcsin (\sqrt{a_L})
    \tag{8}
    \end{align*}
    $$
    """)
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
    mo.md(r"""
    Intuitively, the function $\phi(a_D) = \sin^2(2 \pi a_D)$ oscillates between $0$ and $1$ with a period of $1$, completing a cycle everytime $a_D$ reaches a new integer value. This behaviour mimics the modulo operation from the dyadic map $\mathcal{D}(a_D) = (2 a_D) \mod 1$ which similarly keeps outputs bounded within $[0,1)$ and repeats at each integer boundary.

    We go back and forth between the dyadic and logistic spaces with these key equations

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

    * The first equation says that to transform dyadic space into logistic space, we apply $\phi$ to the dyadic *outputs* $\mathcal{D}^k(a_D)$ and get $\mathcal{L}^k(a_L)$.
    * The second equation says that to transform logistic space into dyadic space, we apply the inverse $\phi^{-1}$ to the *input* $a_L$ before applying the logistic map $\mathcal{L}$ and get $\mathcal{D}^k(a_D)$.

    These equations hold for all iterations $k$ , meaning $\phi$ and $\phi^{-1}$ perfectly relate *every* single point in the dyadic and logistic orbits. Think of these two orbits existing in parallel universes with $\phi$ and $\phi^{-1}$ acting as the bridges between $\mathcal{D}$ and $\mathcal{L}$.
    """)
    return


@app.cell
def _(mo):
    topological_conjugacy_image = mo.image(
        "public/images/topological_conjugacy.png",
        width=400,
        caption="Topological conjugacy between the dyadic and logistic map.",
        style={"display": "block", "margin": "0 auto"}
    )
    topological_conjugacy_image
    return


@app.cell
def _(mo):
    mo.md(r"""
    Previously the dyadic and logistic orbits appeared totally unrelated. But let's now revisit the orbits for $a_D = a_L = 0.431$.

    * Starting from the **dyadic orbit** $(0.431, 0.862, 0.724, 0.448, 0.897, 0.792)$, applying $\phi$ to *after* each dyadic map (eq 10) yields the logistic orbit $(0.431, 0.981, 0.075, 0.277, 0.800, 0.639)$.
    * Starting from the **logistic orbit** $(0.431, 0.981, 0.075, 0.277, 0.800, 0.639)$, applying $\phi^{-1}$ *before* each logistic map (eq 11) yields the dyadic orbit $(0.431, 0.862, 0.724, 0.448, 0.897, 0.792)$.

    Although both these orbits look completely unrelated, they are perfectly connected to one another through $\phi$ and $\phi^{-1}$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## A New Algorithm
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Let's now use the smooth and differentiable logistic map $\mathcal{L}$  as "makeup" to hide the ugly and discontinuous dyadic operation $\mathcal{D}$ in our algorithm. Remember, we still need to be in the dyadic space so our clever bit manipulations will still work out. Here's the strategy:

    1. Encoder: Work in dyadic space where bit manipulation works (use $\phi$) but at the very end output $\alpha$ in logistic space (use $\phi^{-1}$)
    2. Decoder: Work entirely in smooth logistic space using the conjugacy relationship

    This gives us two new beautiful encoder/decoder algorithms where the main changes are bolded:

    /// admonition | **Encoding Algorithm $g(\mathcal{X}, p)$:**

    Given a dataset $\mathcal{X} = \{x_0, ..., x_n\}$ where $x_i \in [0, 1]$ and precision $p$, encode the dataset into $a_L$:

    1. ***Transform data to dyadic coordinates: $z_i = \phi^{-1}(x_i) = \frac{1}{2 \pi} \arcsin⁡( x_i )$ for $i=1, ..., n$***
    2. Convert each transformed number to binary with $p$ bits of precision: $b_i = \text{bin}_p(z_i)$ for $i=1, ..., n$
    3. Concatenate into a single binary string $b = b_0 \oplus  ... \oplus b_n$
    4. Convert to decimal $a_D = \text{dec}(b)$
    5. ***Transform to logistic space: $\alpha = a_L = \phi(a_D) = \sin^2(2 \pi a_D)$***
    6. Return $\alpha$

    ///


    Mathematically, the encoder is defined as

    $$
    \begin{align*}
    \alpha
    &=
    g(p, \mathcal{X}) := \phi \bigg( \text{dec} \Big( \bigoplus_{x_i \in \mathcal{X}} \text{bin}_p(\phi^{-1}(x_i)) \Big) \bigg)
    \end{align*}
    $$

    where $\oplus$ means concatenation. Like before the result is a single, decimal, scalar number $\alpha$ with $np$ bits of precision that contains our entire dataset. However, this time $\alpha$ is in logistic space. We can now discard $\mathcal{X}$ entirely and recover sample $x_i$ by decoding $\alpha$.

    /// admonition | **Decoding Algorithm $f_{\alpha, p}(i)$:**

    Given sample index $i \in \{0, ..., n-1\}$, precision $p$, and the encoded number $\alpha$, recover sample $\tilde{x_i}$:

    1. ***Apply the logistic map $\mathcal{L}$ exactly $ip$ times $\tilde{x}'_i = \mathcal{L}^{ip}(\alpha) = \sin^2 \Big(2^{i p} \arcsin^2(\sqrt{\alpha}) \Big)$***
    2. Extract the first $p$ bits of $\tilde{x}'_i$'s binary representation $b_i = \text{bin}_p(\tilde{x}'_i)$
    3. Convert to decimal $\tilde{x}_i = \text{dec}(b_i)$
    4. Return $\tilde{x}_i$

    ///


    Mathematically, the decoder is defined as

    $$
    \begin{align*}
    \tilde{x}_i
    &=
    f_{\alpha,p}(i)
    :=
    \text{dec} \Big( \text{bin}_p \Big( \mathcal{L}^{ip}(\alpha) \Big) \Big)
    =
    \text{dec} \Big( \text{bin}_p \Big( \sin^2 \Big(2^{ip} \arcsin(\sqrt{\alpha}) \Big) \Big) \Big)
    \end{align*}
    $$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    We've taken the crude, discontinuous dyadic map and transformed it into something smooth and differentiable. The logistic map doesn't *look* like it's doing binary operations, but underneath the elegant trigonometry, it's performing exactly the same bit manipulations as its topological conjugate, the dyadic map. Indeed, the makeup looks pretty great!

    However, nothing is free. The cost of using the logistic map instead of the dyadic map is that our error is now $2 \pi$ times larger,

    $$
    |\tilde{x}_i - x_i | \leq \frac{2 \pi}{2^{p}} = \frac{\pi}{2^{p-1}}
    $$

    We get this $2 \pi$ factor by noting that the derivative of $\phi$ is bounded by $2 \pi$ and applying the mean-value theorem. For a proof, see section 2.5 of [Real numbers, data science and chaos: How to fit any dataset with a single parameter](https://arxiv.org/abs/1904.12320).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
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
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Code Implementation
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Now comes the moment of truth. We've built up all this beautiful math about chaos theory and topological conjugacy, but can we actually code it up?

    If you've been paying attention, there is one crucial implementation detail we have to worry about. If our dataset $\mathcal{X}$ has $n$ samples, each encoded with $p$ bits, $\alpha$ will contain $np$ bits. For ARC-AGI-2 with hundreds of puzzles and high precision, this could be millions of bits. Standard computers can only handle numbers with 32 or 64 bits. How do we even store $\alpha$, much less solve ARC-AGI-2 with it?

    The answer is simple: we can use an arbitrary precision arithmetic library like [mpmath](https://github.com/mpmath/mpmath) that can represent numbers with as many bits as we want. Instead of a regular Python float, we represent $\alpha$ as a mpmath float with $np$ bits of precision. We then run the decoder with mpmath operations and convert the final result back to a regular Python float. However, operations with arbitrary precision arithmetic libraries like mpmath tend to be *significantly* slower than regular floating point operations.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    We implement the logistic encoder

    $$
    \alpha
    =
    g(p, \mathcal{X})
    =
    \phi \bigg( \text{dec} \Big( \bigoplus_{x_i \in \mathcal{X}} \text{bin}_p(\phi^{-1}(x_i)) \Big) \bigg)
    $$

    in code as
    """)
    return


@app.cell
def _(display_fxn, logistic_encoder, mo):
    mo.md(rf"""
    {display_fxn(logistic_encoder)}
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    We compute $\alpha$ in five steps using mpmath with precision set to $np$ bits. Crucially, step 4 produces an mpmath float with the full $np$ bits of precision, which we then transform in step 5 to get our final $np$-bit parameter $\alpha$. Next, we implement the logistic decoder

    $$
    \tilde{x}_i
    =
    f_{\alpha, p}(i)
    =
    \sin^2 \Big(
        2^{x p} \arcsin^2(\sqrt{\alpha})
    \Big)
    $$

    in code as
    """)
    return


@app.cell
def _(display_fxn, logistic_decoder, mo):
    mo.md(rf"""
    {display_fxn(logistic_decoder)}
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Again, we set the mpmath precision to $np$ bits and implement the decoder in a single line using mpmath's arbitrary-precision functions `Sin`, `Arcsin`, and `Sqrt`. That's it. Our entire encoder and decoder, the heart of our one-parameter model, is just a handful of lines and a bit of beautiful mathematics.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    /// details | Our decoder is $f_{\alpha, p}(i) = \text{dec} \Big( \text{bin}_p \Big(\sin^2 \Big( 2^{x p} \arcsin^2(\sqrt{\alpha}) \Big) \Big) \Big)$ but our implementation drops the $\text{dec}(\text{bin}_p(\cdot))$. Why is this okay?
        type: info

    Using mpmath allows us to remove the pesky $\text{dec}(\text{bin}_p(\cdot))$ operations from our decoder

    $$ f_{\alpha, p}(i)
    =
    \text{dec} \Big( \text{bin}_p \Big( \mathcal{L}^{ip}(\alpha) \Big) \Big).
    $$

    In our implementation, we use $\text{dec}(\text{bin}_p(\cdot))$ to truncate $\mathcal{L}^{ip}(\alpha)$ to exactly $p$ bits and then convert $f_{\alpha, p}(i)$ from a $p$-bit mpmath number to a Python float32. During this conversion, Python copies the first $p$ bits of $f_{\alpha, p}(i)$  and then fills the remaining bits of the Python float32 (bits $p+1$ through $32$) with random meaningless junk bits (assuming $p<=32$). Since our model only guarantees accuracy for the first $p$ bits, these random bits don't matter.

    However, converting to binary and back is wildly expensive, especially when $\alpha$ contains millions of bits. Upon taking a closer look, we can, in fact, actually skip the entire $\text{dec}(\text{bin}_p(\cdot))$ step and convert $\mathcal{L}^{ip}(\alpha)$ directly to a Python float32. The first $p$ bits of $\mathcal{L}^{ip}(\alpha)$ still get copied correctly and bits $p+1$ through $32$ get filled with the higher-order bits of $\mathcal{L}^{ip}(\alpha)$ instead of random Python bits. Since our prediction only uses the first $p$ bits, these extra bits are irrelevant whether they come from Python junk or from the higher-order bits of our decoder. Removing $\text{dec}(\text{bin}_p(\cdot))$, our decoder simplifies to exactly what we promised at the start:

    $$ f_{\alpha, p}(i)
    =
    \mathcal{L}^{ip}(\alpha)
    =
    \sin^2 \Big(
        2^{x p} \arcsin^2(\sqrt{\alpha})
    \Big)
    $$

    This is amazing! Usually translating math into code turns beautiful theory into ugly, complicated messes. But surprisingly, leveraging mpmath has the opposite effect and actually makes our decoder even simpler.

    ///
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    To actually run `logistic_encoder` and `logistic_decoder` on ARC-AGI-2, we need three adjustments:

    1. **Adjustment 1: Supervised learning.** ARC-AGI-2 is a supervised problem with input-output pairs $(X,Y)$, but our encoder only handles unsupervised data $(X)$. Solution: ignore the input $X$ and only encode the outputs $Y$ since those are what we need to memorize.
    2. **Adjustment 2: Shape handling.** Our encoder expects scalars, not matrices. Solution: flatten matrices to lists for encoding and reshape back for decoding. For an `m x n` puzzle, we decode `mn` individual elements, running the decoder `mn` times per puzzle, not once. We never explicltly learn the order of inputs.
    3. **Adjustment 3: Data scaling.** ARC-AGI-2 uses integers $0-9$, but our encoder needs values in $[0,1]$. Solution: use a MinMaxScaler to squeeze the data into the right range during encoding and unscale them during decoding. This changes the theoretical error bounds from $|\hat{y} - y| < \frac{\pi}{2^{p-1}}$ to $|\hat{y} - y| < R \Big( \frac{\pi}{2^{p-1}} + \epsilon \Big)$ where $R$ is the range and $\epsilon$ the clipping constant of the MinMaxScaler.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    /// details | Why does MinMaxScaling change our error bound to $|\hat{y} - y| < R \Big( \frac{\pi}{2^{p-1}} + \epsilon \Big)$?
        type: info

    We scale the raw ARC-AGI data $y \in [0, 9]$ to the unit interval $[0,1]$ using the minmax scaler which does a linear shift

    $$
    y_{\text{scaled}} = \frac{y - y_{\min}}{R}
    $$

    and then clips to

    $$
    y_{\text{clipped}} = \text{clip}(y_{\text{scaled}}, \epsilon, 1-\epsilon).
    $$

    Clipping introduces error bounded by:

    $$
    |y_{\text{clipped}} - y_{\text{scaled}}| \leq \epsilon \tag{1}
    $$

    Our one-parameter model predicts $\hat{y}_{\text{scaled}} \in [0,1]$ with decoding error:

    $$
    |\hat{y}_{\text{scaled}} - y_{\text{clipped}}| \leq \frac{\pi}{2^{p-1}} \tag{2}
    $$

    We rescale back to the original domain via $\hat{y} = \hat{y}_{\text{scaled}} \cdot R + y_{\min}$. Therefore, the total error is

    $$
    \begin{align*}
    |\hat{y} - y| &= |(\hat{y}_{\text{scaled}} \cdot R + y_{\min}) - (y_{\text{scaled}} \cdot R + y_{\min})| \\
    &= R |\hat{y}_{\text{scaled}} - y_{\text{scaled}}| \\
    &\leq R (|\hat{y}_{\text{scaled}} - y_{\text{clipped}}| + |y_{\text{clipped}} - y_{\text{scaled}}|) & \text{by the triangle inequality} \\
    &= R \left(\frac{\pi}{2^{p-1}} + \epsilon\right) & \text{by equations (1) and (2)}
    \end{align*}
    $$
    ///
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    With these adjustments, we are now ready to create the final implementation of the one-parameter model. The code is quite simple and looks like a standard scikit-learn ML model:

    * `model.fit` runs the encoder. It also scales and reshapes the data.
    * `model.predict` runs the decoder. It also reverses the data scaling and reshaping.
    * `model.verify` checks that the outputted predictions are within the theoretical error bounds we derived.
    """)
    return


@app.cell
def _(OneParameterModel, display_fxn, mo):
    mo.md(rf"""
    {display_fxn(OneParameterModel)}
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    This is the entire one-parameter model, short and sweet! Since arbitrary precision arithmetic operations are so slow, we implemement three speed ups for the decoder.

    1. **Parallelization:** Because each number is decoded independently, we can decode all number in parallel with `multiprocessing.Pool`. This speeds up the for loop over the indices `range(len(y_scaled))`.
    2. **Precomputation:** Calculate `arcsin(sqrt(alpha))` once before decoding instead of recomputing it every time we call `logistic_decoder`. This eliminates repeated expensive trigonometric and square root operations on huge $np$-bit numbers like $\alpha$.
    3. **Adaptive precision:** We currently use all $np$ bits of $\alpha$ every decoding step, i.e. we set `mp.prec = full_precision`. However, in the $i$th decoding step, we only need the first $p(i+1)+1$ bits of $\alpha$. This allows us to work with fewer bits, drastically reducing the computation needed at each step, and makes everything faster.

    These speedups are gated behind the `fast` flag and are turned on by default. On my M1 Macbook, they give a 10x speedup.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    /// details | How does adaptive precision work? Why can we use $p(i+1)+1$ bits instead of $np$ bits in the $i$th decoding step?
        type: info

    Each sample is encoded in $p$ bits, so the $i$th sample occupies bits $ip$ through $ip + (p-1) = p(i+1) - 1$ of $\alpha$. The parts of $\alpha$ beyond $\alpha$ beyond $p(i+1) - 1$ bits are irrelevant in iteration $i$.

    By setting mpmath's precision to exactly $p(i+1) - 1$ bits in iteration $i$, we perform computation on fewer bits, increasing the precision gradually: $p$ bits in iteration $0$, $2p$ bits in iteration $1$, and so on, up to $np$ bits in the final iteration. This reduces the total arithmetic cost from $n \cdot (np)$ bit-operations to

    $$
    p(1+2+...+n) = \frac{n(n+1)}{2} p,
    $$

    which is roughly 2x fewer arithmetic operations. Theoretically this is a constant factor improvement. However, in practice this yields a dramatic speedup in mpmath.

    A key important caveat is that this optimization only works in dyadic space where the bit structure is explicit. In logistic space, the bit positions are scrambled, making reduced precision unusable. For this reason, we apply reduced precision only after $\phi^{-1}$ transforms the value into dyadic space. Shout out to Claude for helping me to debug this nuanced point!

    Finally, to improve numerical stability, we set mpmath's precision to $p(i+1)+1$ bits -- two bits higher than the normal $p(i+1)-1$. These two extra bits are not for extracting additional information from $\alpha$. Instead, they act as a numerical buffer that helps preserves the accuracy of mpmath’s arithmetic. Empirically, we need this otherwise mpmath does not work properly. I'm not sure why...

    ///
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Now let's try out the one-parameter model and encode the entire ARC-AGI-2 public eval set (120 puzlzes) into $\alpha$ using precision $p=6$.
    """)
    return


@app.cell
def _(ds, pad_arc_agi_2):
    X, y = pad_arc_agi_2(ds['eval'])
    return X, y


@app.cell
def _(OneParameterModel, X, mo, y):
    model1 = OneParameterModel(precision=6)
    model1.fit(X, y)
    mo.show_code()
    return (model1,)


@app.cell
def _(mo):
    mo.md(r"""
    This is our one-parameter model in its full glory! This scalar $\alpha$ is is 130,044 digits long and all we need to correctly predict the question output of any ARC-AGI-2 public eval set puzzle!
    """)
    return


@app.cell
def _(display_alpha, model1):
    p1 = 6
    alpha1_str = str(model1.alpha)
    display_alpha(p1, alpha1_str)
    return alpha1_str, p1


@app.cell
def _(mo):
    mo.md(r"""
    For instance, consider puzzle 23 of the public eval set. In `model.fit()`, we ignore $X$ which contains the 3 examples and the question input
    """)
    return


@app.cell
def _():
    idx = 23
    return (idx,)


@app.cell
def _(ds, idx, plot_arcagi):
    plot_arcagi(ds, "eval", idx, hide_question_output=True, size=2.5) # 17, 23, 26, 27
    return


@app.cell
def _(mo):
    mo.md(r"""
    and only encode the question output $Y$
    """)
    return


@app.cell
def _(ds, idx, plot_question):
    plot_question(ds, 'eval', idx, io='output', size=4)
    return


@app.cell
def _(mo):
    mo.md(r"""
    into $\alpha$, along with the question outputs from all the other 120 public eval puzzles. To recover the question output of puzzle 23, we run `model.predict()` and decode $\alpha$
    """)
    return


@app.cell
def _(mo, model1, np):
    y1_pred = model1.predict(np.array([23]))

    mo.show_code()
    return (y1_pred,)


@app.cell
def _(mo):
    mo.md(r"""
    The first row of the predictions looks like this numerically
    """)
    return


@app.cell
def _(y1_pred):
    print(f'y1_pred[0] = {y1_pred[0, 0]}')
    return


@app.cell
def _(mo):
    mo.md(r"""
    and visually look like
    """)
    return


@app.cell
def _(np, plot_matrix, plt):
    def plot_prediction(ds, split, i, predictions=None, precisions=None, alpha_n_digits=None, size=2.5, w=0.9, show_nums=False):
      puzzle = ds[split][i]
      nq = 1 # len(puzzle['question_inputs'])
      n_pred = len(predictions) if predictions is not None else 0
      mosaic = [[f'Q.{j+1}_out' for j in range(nq)] + [f'pred_{k}' for k in range(n_pred)]]
      fig, axes = plt.subplot_mosaic(mosaic, figsize=(size*(nq+n_pred)+3, 5))
      plt.suptitle(f'ARC-AGI-2 {split.capitalize()} puzzle #{i}', fontsize=18, fontweight='bold', y=0.98)

      for j in range(nq):
        plot_matrix(puzzle['question_outputs'][j], axes[f'Q.{j+1}_out'], title=f"Q.{j+1} Output", status='predict', w=w, show_nums=show_nums)

      if n_pred:
        if precisions is None: precisions = [None]*n_pred
        if alpha_n_digits is None: alpha_n_digits = [None]*n_pred
        for k in range(n_pred):
          pred = np.array(predictions[k])[:len(puzzle['question_outputs'][0]), :len(puzzle['question_outputs'][0][0])]
          title = "Q.1 Prediction"
          if alpha_n_digits[k] is not None: title = f"len(α)={alpha_n_digits[k]:,} digits\n\n{title}"
          if precisions[k] is not None: title = f"Precision={precisions[k]}\n{title}"
          plot_matrix(pred, axes[f'pred_{k}'], title=title, w=w, show_nums=show_nums)
        fig.add_artist(plt.Line2D([nq/(nq+n_pred), nq/(nq+n_pred)], [0.05, 0.87], color='#333333', linewidth=5, transform=fig.transFigure))
        fig.text(nq/(2*(nq+n_pred)), 0.91, 'Questions', ha='center', va='top', fontsize=13, fontweight='bold', color='#444444', transform=fig.transFigure)
        fig.text((nq+n_pred/2)/(nq+n_pred), 0.91, 'Predictions', ha='center', va='top', fontsize=13, fontweight='bold', color='#444444', transform=fig.transFigure)
      else:
        fig.text(0.5, 0.91, 'Questions', ha='center', va='top', fontsize=13, fontweight='bold', color='#444444', transform=fig.transFigure)

      fig.patch.set_linewidth(5)
      fig.patch.set_edgecolor('#333333')
      fig.patch.set_facecolor('#eeeeee')
      plt.tight_layout(rect=[0, 0, 1, 0.94], h_pad=1.0)
      return fig

    return (plot_prediction,)


@app.cell
def _(alpha1_str, ds, idx, p1, plot_prediction, y1_pred):
    plot_prediction(ds, "eval", idx, [y1_pred.squeeze()], [p1], [len(alpha1_str)], show_nums=True, size=2)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Here, the correct answer is on the left and the one-parameter model's prediction is on the right. (Remember, the colors are just for display purposes, the model really sees numerical values.) Taking a closer look

    * The green cells equal 3, and our predictions came close, spanning 2.9, 2.8, and 3.0.
    * The yellow cells equal 4, but we predicted 3.7.
    * The light blue cells equal 8 but we predicted 7.8 and 7.7.

    Across the entire image, our predictions are close to the correct values, typically off by only a small fraction. What went wrong?

    These small errors happen because of our precision setting $p$. Remember, the encoder saves each number using only $p$ bits and throws away everything else. This cuttoff creates quantization errors up to

    $$
    R \Big( \frac{\pi}{2^{p-1}} + \epsilon \Big) = 0.88
    $$

    where precision $p=6$, MinMaxScaler range $R=9$, and MinMaxScaler clipping factor $\epsilon=10^{-12}$. All our errors are indeed less than $0.88$, meaning our theoretical bound holds true in practice. There's nothing broken here. This is just what happens when you use finite precision.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    If we run the one-parameter model with $p=14$ and $p=4$
    """)
    return


@app.cell
def _(OneParameterModel, mo, np):
    @mo.persistent_cache
    def run_model(X, y, p, idx):
        # fit and predict a model
        model2 = OneParameterModel(precision=p).fit(X, y)
        alpha2_str = str(model2.alpha)
        y2_pred = model2.predict(np.array([idx]))
        return model2, alpha2_str, y2_pred

    return (run_model,)


@app.cell
def _(X, idx, run_model, y):
    p2 = 14
    model2, alpha2_str, y2_pred = run_model(X, y, p2, idx)
    return alpha2_str, model2, p2, y2_pred


@app.cell
def _(X, idx, run_model, y):
    p3 = 4
    model3, alpha3_str, y3_pred = run_model(X, y, p3, idx)
    return alpha3_str, p3, y3_pred


@app.cell
def _():
    # display_alpha(p2, alpha2_str)
    return


@app.cell
def _():
    # display_alpha(p3, alpha3_str)
    return


@app.cell
def _(
    alpha1_str,
    alpha2_str,
    alpha3_str,
    ds,
    idx,
    p1,
    p2,
    p3,
    plot_prediction,
    y1_pred,
    y2_pred,
    y3_pred,
):
    plot_prediction(ds, "eval", idx, [y3_pred.squeeze(), y1_pred.squeeze(), y2_pred.squeeze()], [p3, p1, p2], [len(alpha3_str.strip('0.')), len(alpha1_str.strip('0.')), len(alpha2_str.strip('0.'))], show_nums=True, size=2.8)
    return


@app.cell
def _(mo):
    mo.md(r"""
    With $p=4$ we get degenerate results. But with $p=14$ every prediction is exactly right. But there's a tradeoff: we need more space. For $p=14$ $\alpha$ has 455,155 digits whereas it only needs 195,066 digits for $p=6$. The higher we set $p$, the more accurate our encoding becomes, but the more memory it requires.
    """)
    return


@app.cell
def _(idx, mo, y, y2_pred):
    mo.md(rf"""
    Taking a closer look at our $p=14$ predictions, they're not perfectly accurate—they only match the ground truth to about 2 decimal places (assuming rounding):

    ```py
    y_pred[0, 0] = {y2_pred[0, 0, 0]}
    ```

    ```py
    y[0, 0] = {y[idx, 0, 0]}
    ```

    In binary,
    """)
    return


@app.function
def diff(a, b, name_a="", name_b=""):
    from IPython.display import HTML, display
    def line(s, t): return ''.join(f'<span style="color:{"green" if x==y else "red"}">{x}</span>  ' for x, y in zip(s, t))
    nums = ''.join(f'{i:<3}' for i in range(1, len(a)+1))
    pad = max(len(name_a), len(name_b))
    display(HTML(f'<div style="white-space:pre; font-family:monospace">{"":<{pad}} {nums}<br>{name_a:>{pad}} {line(a, b)}<br>{name_b:>{pad}} {line(b, a)}</div>'))


@app.cell
def _(decimal_to_binary, idx, model1, y, y1_pred):
    diff(decimal_to_binary(model1.scaler.transform(y[idx, 0, 0]), 32), decimal_to_binary(model1.scaler.transform(y1_pred[0, 0, 0]), 32))
    return


@app.cell
def _(decimal_to_binary, idx, model2, y, y2_pred):
    diff(decimal_to_binary(model2.scaler.transform(y[idx, 0, 0]), 32), decimal_to_binary(model2.scaler.transform(y2_pred[0, 0, 0]), 32))
    return


@app.cell
def _(decimal_to_binary, idx, model4, y, y4_pred):
    diff(decimal_to_binary(model4.scaler.transform(y[idx, 0, 0]), 32), decimal_to_binary(model4.scaler.transform(y4_pred[0, 0, 0]), 32))
    return


@app.cell
def _(math):
    R = 9
    p_star = math.ceil(33 + math.log2(R * math.pi))
    print(p_star)
    return


@app.cell
def _(mo):
    mo.md(r"""
    This raises the question: what precision $p$ do we need to store each sample in for the one-parameter model to be accurate up to $p^*=32$ bits? In other words, we want

    $$
    \begin{align*}
    |\hat{y} - y|
    & \leq
    2^{p^*}
    \\
    \end{align*}
    $$

    which means we need the upper bound of our error to be less than $2^{p^*}$

    $$
    \begin{align*}
    R \left(\frac{\pi}{2^{p-1}} + \epsilon\right)
    & \leq
    2^{p^*}
    \end{align*}
    $$

    Plugging in $R=9$ and $\epsilon=1e-12$ and solving for $p$

    $$
    \begin{align*}
    p
    &\geq
    \Bigg \lceil
    \log_2 \left( \frac{\pi \cdot R \cdot 2^{p^*}}{1 - \epsilon \cdot R \cdot 2^{p^*}} \right) + 1
    \Bigg \rceil
    \\
    & \geq
    \lceil 37.87 \rceil
    \\
    p
    & \geq
    38
    \end{align*}
    $$

    Each number must be encoded into $\alpha$ with 38 bits of precision for the prediction to be accurate up to 32 bits. This means each of the $900$ numbers in our $30 \times 30$ image needs $p=38$ bits of precision. For all $n=120$ eval puzzles, we need

    $$
    900 \times 120 \times 38 = 4{,}104{,}000 \text{ bits} \approx 0.513 \text{ MB}
    $$

    In decimal notation, $\alpha$ must store approximately **1,235,427 digits**.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    /// details | **Are there any constraints on $\epsilon$?**

    The $\epsilon$ used for the MinMaxScaler clipping must be small enough that the clipping noise doesn't mask the significant bits of the data. This constraint is reflected in the denominator of our logarithmic term:

    $$1 - \epsilon \cdot R \cdot 2^{p^*}$$

    We need this term to be positive as you cannot take the logarithm of a non-positive number. If the target precision $p^*$ is really big, we must have a smaller clipping noise $\epsilon$ must be small enough to counteract the massive scaling of $2^{p^*}$. If $\epsilon$ is too large, clipping error destroys the signal, making $p^*$ bits of accuracy mathematically impossible.

    ///
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    `OneParameterModel` itself is ~50 lines of code and the math functions it uses are probably another ~50 lines of code. Only around 100 lines to get a perfect score on ARC-AGI-2!

    Let's train the one-parameter model on all 400 puzzles of ARC-AGI-2.
    """)
    return


@app.cell
def _(X, idx, run_model, y):
    # actually fit + predict the one parameter model here so it is cached
    # the code below is in markdown and is just for display purposes
    p4 = 38
    model4, alpha4_str, y4_pred = run_model(X, y, p4, idx)
    return alpha4_str, model4, p4, y4_pred


@app.cell
def _(mo):
    mo.md(r"""
    ```py
    model4 = OneParameterModel(precision=38)
    model4.fit(X, y)
    alpha4_str = str(model4.alpha)
    ```
    """)
    return


@app.cell
def _(alpha4_str, display_alpha, p4):
    display_alpha(p4, alpha4_str)
    return


@app.cell
def _(mo):
    mo.md(r"""
    This is our perfect one-parameter model! Let's take another look at our prediction for the first puzzle of the public eval dataset.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ```py
    y4_pred = model4.predict(np.array([idx]))
    model4.verify(y4_pred, y[idx])
    ```
    """)
    return


@app.cell
def _(alpha4_str, ds, idx, p4, plot_prediction, y4_pred):
    plot_prediction(ds, "eval", idx, [y4_pred.squeeze()], [p4], [len(alpha4_str.strip('0.'))], size=3, show_nums=True)
    return


@app.cell
def _(idx, mo, y, y4_pred):
    mo.md(rf"""
    With $p=38$ our predictions perfectly match the ground truth for all 32 bits.

    ```py
    {y4_pred[0, 0, 0]=}
    ```

    ```py
    {y[idx, 0, 0]=}
    ```
    """)
    return


@app.cell
def _(decimal_to_binary, model4, y4_pred):
    y5_pred_binary = decimal_to_binary(model4.scaler.transform(y4_pred[0, 0, 0]), 32)
    y5_pred_binary
    return (y5_pred_binary,)


@app.cell
def _(decimal_to_binary, idx, model4, y):
    y_binary = decimal_to_binary(model4.scaler.transform(y[idx, 0, 0]), 32)
    y_binary
    return (y_binary,)


@app.cell
def _(y5_pred_binary, y_binary):
    assert y5_pred_binary == y_binary, f'{y5_pred_binary=}\n{y_binary=}'
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Bytes on Bytes
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Let's be honest: claiming this is a "one-parameter" model is, in some sense, cheating. The model simply hides its complexity in an immense number of digits rather than its parameter count. Instead of millions of parameters, it has millions of digits.

    A better way to measure model size is by the total bytes of the model weights (number of parameters × bytes per parameter). This accounts for both the quantity of parameters and the precision at which each is stored, capturing the true information content required to specify the model.

    [Scraping](https://arcprize.org/media/data/leaderboard/evaluations.json) the official ARC-AGI leaderboard, we can plot each model's size in bytes against their ARC-AGI-2 scores. (Let's assume all frontier labs' closed source models have 1T parameters and are in fp16.)
    """)
    return


@app.cell
def _(Path, json, pd):
    import requests


    def load_arc_evals(path="public/data/arc-agi-evaluations.json"):
        path = Path(path)

        if not path.exists():
            url = "https://arcprize.org/media/data/leaderboard/evaluations.json"
            path.write_text(json.dumps(requests.get(url).json()))

        df = pd.read_json(path)
        df["version"] = df["datasetId"].str.extract(r"(v\d+)")
        df["eval"] = df["datasetId"].str.extract(r"(Public|Semi_Private|Private)")

        df = (
            df.pivot_table(
                index=["modelId", "version"], columns="eval", values="score"
            )
            .reset_index()
            .rename(
                columns={
                    "modelId": "model",
                    "Public": "public eval score",
                    "Semi_Private": "semi-private eval score",
                }
            )
        )

        df[["public eval score", "semi-private eval score"]] *= 100
        df = (
            df[df["version"] == "v2"]
            .dropna(subset=["public eval score", "semi-private eval score"])
            .drop(columns=["Private", "version"])
        )

        models = {
            "gpt-5": {"pattern": r"^gpt-5", "params": 1_000_000_000_000},
            "gemini-3": {"pattern": r"^gemini-3", "params": 1_000_000_000_000},
            "claude-4-5": {
                "pattern": r"claude-.*-4-5",
                "params": 1_000_000_000_000,
            },
            "grok-4": {"pattern": r"^grok-4", "params": 1_000_000_000_000},
            # "o4-mini": {"pattern": r"^o4-mini", "params": 1_000_000_000_000},
            "qwen3-235b": {"pattern": r"^qwen3-235b", "params": 235_000_000_000},
            "R1": {"pattern": r"^R1", "params": 671_000_000_000},
            "ARChitects": {"pattern": r"^ARChitects$", "params": 8_000_000_000},
            "trm": {"pattern": r"^trm", "params": 7_000_000},
            "hrm": {"pattern": r"^hrm", "params": 27_000_000},
        }

        rows = []
        for m in models.values():
            sub = df[df["model"].str.contains(m["pattern"], regex=True)]
            if sub.empty:
                continue
            best = sub.loc[sub["public eval score"].idxmax()].copy()
            best["# of parameters"] = m["params"]
            best["weight dtype"] = "fp16"
            best["weight bytes"] = m["params"] * 2
            rows.append(best)

        df = pd.DataFrame(rows)

        # add one-parameter model row
        df.loc[len(df)] = {
            "model": "one-parameter model",
            "public eval score": 100.0,
            "semi-private eval score": 0.0,
            "# of parameters": 1,
            "weight dtype": "arbitrary",
            "weight bytes": 4_000_000,
        }

        # sort by weight bytes and reset index
        df = df.sort_values('weight bytes')
        df = df.reset_index(drop=True)
        return df

    return (load_arc_evals,)


@app.cell
def _(load_arc_evals):
    df = load_arc_evals()
    df
    return (df,)


@app.cell
def _(np, plt):
    import matplotlib
    def draw_curly_brace(ax, x1, x2, y, text):
        """Adds a smooth horizontal curly brace with a central tip."""
        # Scale parameters for the 'curl' of the brace
        height = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.03

        # We define the brace using 4 Bezier curves
        # Points: start, left-arch, center-tip, right-arch, end
        mid = np.sqrt(x1 * x2) if ax.get_xscale() == 'log' else (x1 + x2) / 2

        # Path codes for Matplotlib (MoveTo, Curve4, Curve4...)
        # This creates the characteristic { shape horizontally
        cmds = [matplotlib.path.Path.MOVETO, matplotlib.path.Path.CURVE4, matplotlib.path.Path.CURVE4, matplotlib.path.Path.CURVE4, 
                matplotlib.path.Path.CURVE4, matplotlib.path.Path.CURVE4, matplotlib.path.Path.CURVE4]

        # Vertices: Using log space for X if necessary to keep it looking symmetrical
        # Note: For log scales, we interpolate in log10 space
        lx1, lx2, lmid = np.log10(x1), np.log10(x2), np.log10(mid)
        l_q1 = lx1 + (lmid - lx1) * 0.5
        l_q3 = lmid + (lx2 - lmid) * 0.5

        verts = [
            (x1, y), # Start
            (x1, y + height), (10**l_q1, y), (mid, y + height), # Left half
            (10**l_q3, y), (x2, y + height), (x2, y) # Right half
        ]

        path = matplotlib.path.Path(verts, cmds)
        patch = matplotlib.patches.PathPatch(path, facecolor='none', lw=2, edgecolor='black', clip_on=False)
        ax.add_patch(patch)

        # Add the text above the tip
        ax.text(mid, y + height * 1.5, text, ha='center', va='bottom', 
                fontsize=14, fontweight='bold', color='black')

    def plot_efficiency(df, score_key):
        def fmt(b):
            for u in ['B','KB','MB','GB','TB']:
                if b < 1024: return f"{b:.1f}{u}"
                b /= 1024

        fig, ax = plt.subplots(figsize=(10, 7))
        colors = plt.cm.tab20(np.linspace(0, 1, len(df)))

        for i, (idx, r) in enumerate(df.iterrows()):
            is_target = r['model'] == 'one-parameter model'
            ax.scatter(r['weight bytes'], r[score_key], 
                       color=colors[i], 
                       s=1000 if is_target else 400, 
                       alpha=1.0 if is_target else 0.4, 
                       edgecolors='black', 
                       linewidths=3 if is_target else 1,
                       marker='*' if is_target else 'o',
                       zorder=3 if is_target else 2)

        ax.set_xscale('log')

        # Add labels
        for _, r in df.iterrows():
            is_target = r['model'] == 'one-parameter model'
            ha = 'left' if is_target or r['model'] == 'trm-2025-10-07' else 'right'
            ax.annotate(f" {r['model']}", (r['weight bytes'], r[score_key]), 
                        fontsize=18 if is_target else 12, 
                        fontweight='bold' if is_target else 'normal',
                        va='center', ha=ha,
                        xytext=(10, 0) if ha == 'left' else (-10, 0),
                        textcoords='offset points')

        # --- Draw the Curly Brace ---
        try:
            m1 = df[df['model'] == 'one-parameter model'].iloc[0]
            m2 = df[df['model'] == 'gpt-5-2-2025-12-11-thinking-xhigh'].iloc[0]

            # Position brace slightly above the higher of the two points
            y_max = max(m1[score_key], m2[score_key])
            brace_y = y_max + (ax.get_ylim()[1] * 0.08)

            draw_curly_brace(ax, m1['weight bytes'], m2['weight bytes'], brace_y, "to genuinely learn")
        except Exception as e:
            raise e

        # Formatting
        ticks = 10**np.arange(np.floor(np.log10(df['weight bytes'].min())), 
                              np.ceil(np.log10(df['weight bytes'].max())) + 1)
        ax.set_xticks(ticks)
        ax.set_xticklabels([fmt(t) for t in ticks])
        ax.set(title=f"Model Efficiency: ARC-AGI-2", xlabel="Model Size (Bytes)", ylabel="Score")

        plt.tight_layout()
        return fig

    return


@app.cell
def _(plt):
    def plot_efficiency_twitter(df, score_key, ax=None, marker='o', 
                               label_suffix="", show_labels=True, save_path=None):
        """Clean, minimal Pareto chart optimized for Twitter with refined layering."""

        def fmt(b):
            if b >= 1e12: return f"{int(b/1e12)} TB"
            if b >= 1e9: return f"{int(b/1e9)} GB"
            if b >= 1e6: return f"{int(b/1e6)} MB"
            if b >= 1e3: return f"{int(b/1e3)} KB"
            return f"{int(b)} B"

        display_names = {
            'one-parameter model': 'one-parameter model',
            'gpt-5-2-2025-12-11-thinking-xhigh': 'GPT-5',
            'gemini-3-deep-think-preview': 'Gemini-3',
            'claude-sonnet-4-5-20250514': 'Claude-4.5',
            'grok-4-0125': 'Grok-4',
            'qwen3-235b-a22b-fp8': 'Qwen3-235B',
            'R1': 'DeepSeek-R1',
            'ARChitects': 'ARChitects',
            'trm-2025-10-07': 'TRM',
            'hrm-arc1-2025-10-07': 'HRM',
        }

        GOLD = '#FFD700'
        GRAY = '#6B7280'

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')
        else:
            fig = ax.get_figure()

        ax.set_facecolor('white')

        # Plot points
        for _, r in df.iterrows():
            is_target = r['model'] == 'one-parameter model'

            ax.scatter(
                r['weight bytes'],
                r[score_key],
                color=GOLD if is_target else GRAY,
                s=800 if is_target else 250,
                alpha=1.0 if is_target else 0.5,
                edgecolors='#1a1a1a' if is_target else 'none',
                linewidths=2.5 if is_target else 0,
                marker='*' if is_target else marker,
                zorder=10 if is_target else 2
            )

        # Plot labels only if enabled
        if show_labels:
            for _, r in df.iterrows():
                is_target = r['model'] == 'one-parameter model'
                name = display_names.get(r['model'], r['model'])
                full_label = f"{name}{label_suffix}"

                # Smart positioning based on model type
                if is_target:
                    ha, va = 'left', 'center'
                    offset = (20, 0)
                elif 'gpt' in r['model'].lower():
                    ha, va = 'right', 'bottom'
                    offset = (-10, 8)
                elif 'gemini' in r['model'].lower():
                    ha, va = 'left', 'top'
                    offset = (10, -8)
                else:
                    ha, va = 'right', 'center'
                    offset = (-15, 0)

                ax.annotate(
                    full_label,
                    (r['weight bytes'], r[score_key]),
                    fontsize=14 if is_target else 9,
                    fontweight='bold' if is_target else 'normal',
                    color='#1a1a1a' if is_target else '#4a4a4a',
                    va=va, ha=ha,
                    xytext=offset,
                    textcoords='offset points',
                    # Add background box to text for readability when overlapping
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='none', alpha=0.6)
                )

        # Global Styling
        ax.set_xscale('log')
        ax.grid(True, axis='y', linestyle='-', alpha=0.15, color='#888888')
        ax.grid(True, axis='x', linestyle='-', alpha=0.08, color='#888888')

        # Set limits slightly wider to accommodate labels
        ax.set_xlim(8e5, 4e12)
        ax.set_ylim(-10, 110)

        ticks = [1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12]
        ax.set_xticks(ticks)
        ax.set_xticklabels([fmt(t) for t in ticks], fontsize=11)
        ax.set_yticklabels([f'{int(y)}%' for y in ax.get_yticks()], fontsize=11)

        ax.set_xlabel('Model Size (Bytes)', fontsize=12, fontweight='medium', color='#333333')
        ax.set_ylabel('Evaluation Score (%)', fontsize=12, fontweight='medium', color='#333333')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cccccc')
        ax.spines['bottom'].set_color('#cccccc')

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

        return ax

    return (plot_efficiency_twitter,)


@app.cell
def _():
    # # Create one shared figure
    # fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')

    # # 1. Plot Dataset 1 (Public) - SHOW LABELS
    # plot_efficiency_twitter(df, 'public eval score', ax=ax, marker='o', 
    #                        label_suffix=" (Pub)", show_labels=True)

    # # 2. Plot Dataset 2 (Semi-Private) - HIDE LABELS
    # plot_efficiency_twitter(df, 'semi-private eval score', ax=ax, marker='s', 
    #                        label_suffix="", show_labels=False)

    # ax.set_title("ARC-AGI-2 Efficiency: Public & Semi-Private", fontsize=16)
    # plt.show()
    return


@app.cell
def _(df, plot_efficiency_twitter):
    plot_efficiency_twitter(df, 'public eval score', save_path='public/images/pareto_public_eval.png')
    return


@app.cell
def _(mo):
    mo.md(r"""
    The one-parameter model is Pareto optimal on the public eval set of ARC-AGI-2! It doesn't just achieve the highest score for its size, it achieves the highest score out of all the models. Remember, we measure the model size in total bytes so this captures both the number of parameters and the bytes of each parameter.

    But the one-parameter model also gets a big fat zero on the semi-private eval set. We can plot
    """)
    return


@app.cell
def _(df, plot_efficiency_twitter):
    plot_efficiency_twitter(df, 'semi-private eval score', save_path='public/images/pareto_semi_private.png')
    return


@app.cell
def _(mo):
    mo.md(r"""
    While the one-parameter model achieves a perfect 100% on the public ARC-AGI-2 evaluation set, its performance collapses to zero on semi-private evaluation set. Every other model maintains a consistent score across public and semi-private sets, but the one-parameter model’s failure on held-out data reveals that it has overfit and is nothing more than a clever mathematical hack.

    Taking a closer look, most top-tier models approach 1TB in size, while the one-parameter model is a mere 5MB. If 5MB can literally memorize the entire ARC-AGI-2 dataset, what is the purpose of those extra trillion bytes in larger models?

    The answer is generalization.

    A model is much more than a compressed form of its training data. True pattern recognition requires "excess" bytes to navigate the space between training samples and generalize to unseen examples. These extra bytes allow for language, reasoning, and pattern matching. It is in these excess bytes that true intelligence lies.
    """)
    return


@app.cell
def _(np, plt):
    from matplotlib.lines import Line2D

    def plot_dual_efficiency_bars(df, score_col_1, score_col_2, label_1='Public Eval', label_2='Semi-Private Eval', save_path=None):
        # 1. Setup & Sorting
        d = df.sort_values('weight bytes').reset_index(drop=True)

        # Size Annotation Helper (Formatted for inside labels)
        def fmt_size(b):
            if b >= 1e12: return f"{int(b/1e12)}TB"
            if b >= 1e9: return f"{int(b/1e9)}GB"
            if b >= 1e6: return f"{int(b/1e6)}MB"
            if b >= 1e3: return f"{int(b/1e3)}KB"
            return f"{int(b)}B"

        # 2. Collision Avoidance Layout
        raw_log_pos = np.log10(d['weight bytes'])
        group_width, min_gap = 0.28, 0.45 

        final_positions = []
        last_pos = -1000
        for raw_pos in raw_log_pos:
            current_pos = max(raw_pos, last_pos + min_gap)
            final_positions.append(current_pos)
            last_pos = current_pos
        final_positions = np.array(final_positions)

        # 3. Plotting Setup
        fig, ax = plt.subplots(figsize=(15, 9), facecolor='white')
        ax.set_facecolor('white')
        bar_w = group_width / 2.2

        # Color Palette
        MY_BLUE_DARK, MY_BLUE_LIGHT = '#1E3A8A', '#60A5FA'
        COMP_GRAY_DARK, COMP_GRAY_LIGHT = '#374151', '#D1D5DB'

        # 4. Draw Bars & Labels
        for i, row in d.iterrows():
            is_target = (row['model'] == 'one-parameter model')
            c1 = MY_BLUE_DARK if is_target else COMP_GRAY_DARK
            c2 = MY_BLUE_LIGHT if is_target else COMP_GRAY_LIGHT

            # Public Bar
            h1 = row[score_col_1]
            b1 = ax.bar(final_positions[i] - bar_w/2, max(h1, 0.8), width=bar_w, 
                        color=c1, alpha=1.0, zorder=3)
            ax.bar_label(b1, labels=[f"{int(h1)}"], padding=5, fontsize=14, 
                        fontweight='bold', color=c1)

            # Semi-Private Bar
            h2 = row[score_col_2]
            b2 = ax.bar(final_positions[i] + bar_w/2, max(h2, 0.8), width=bar_w, 
                        color=c2, alpha=1.0, zorder=3)
            ax.bar_label(b2, labels=[f"{int(h2)}"], padding=5, fontsize=14, 
                        fontweight='bold', color=c2)

        # 5. Combined X-Axis Labels (Model Name + Size)
        tick_labels = []
        for _, row in d.iterrows():
            name = 'One-Parameter Model' if row['model'] == 'one-parameter model' else row['model']
            size_str = fmt_size(row['weight bytes'])
            # Creates a 2-line label
            tick_labels.append(f"{name}\n({size_str})")

        ax.set_xticks(final_positions)
        ax.set_xticklabels(tick_labels, rotation=10, ha='center', fontsize=11, fontweight='500')

        # Color the One-Parameter Model label specifically
        for i, label in enumerate(ax.get_xticklabels()):
            if 'One-Parameter Model' in label.get_text():
                label.set_color(MY_BLUE_DARK)
                label.set_fontweight('bold')
            else:
                label.set_color('#4B5563')

        # 6. Global Styling
        ax.set_title("Public vs. Semi-Private ARC-AGI-2 Scores", 
                     fontsize=24, fontweight='bold', color='#111827', pad=30)
        ax.set_ylabel('Score (%)', fontsize=14, fontweight='bold', labelpad=10)
        ax.set_xlabel('Model Size (Bytes)', fontsize=14, fontweight='bold', labelpad=20)

        ax.set_ylim(0, 115) 
        ax.grid(axis='y', linestyle='--', alpha=0.3, color='#94a3b8', zorder=0)

        for s in ['top', 'right', 'bottom']: ax.spines[s].set_visible(False)
        ax.spines['left'].set_color('#E5E7EB')

        # Legend
        legend_elements = [
            Line2D([0], [0], color=COMP_GRAY_DARK, lw=8, label=label_1),
            Line2D([0], [0], color=COMP_GRAY_LIGHT, lw=8, label=label_2)
        ]
        ax.legend(handles=legend_elements, frameon=False, loc='upper right', 
                  fontsize=12, handletextpad=0.5)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    return


@app.cell
def _():
    # plot_dual_efficiency_bars(
    #     df, 
    #     score_col_1='public eval score', 
    #     score_col_2='semi-private eval score',
    #     label_1='Public Score',
    #     label_2='Semi-Private Score'
    # )
    # plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Conclusion

    > "When a measure becomes a target, it ceases to be a good measure" - Charles Goodhart
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Training on test.**

    Despite never using the training set, performing no pretraining, and having only one parameter, the one-parameter model gets 100% on the public eval set of ARC-AGI-2. It takes the idea of "training on test" to the extreme and encodes the question outputs of the entire public eval set directly into $\alpha$, achieving 100% accuracy while learning nothing. Simply shuffling the dataset will cause this model to break down as the decoder depends on the index $i$, not the sample $x_i$. It should be abundantly clear that the one-parameter model has no ability to generalize whatsoever. It would get a 0% on the private, heldout eval set of ARCI-AGI-2.

    The one-parameter model is utterly impractical and, frankly, an absurd hack. But that's precisely the point: it is absurd to train on the test set just to get to the top of a leaderboard.

    Yet this is exactly what occurs in the AI community.

    Top AI labs quietly train on their test sets. It is rumoured these labs have entire teams who generate synthetic dataset for the sole purpose of succeeding on a specific benchmark. I've also heard that at certain labs, your pay is based on getting a particular score on a particular benchmark. Though it is important to incentivize progress, behavior like this can create a culture of benchmark maxing.

    <blockquote class="twitter-tweet"><p lang="en" dir="ltr">dude you&#39;re not supposed to train on the train set that&#39;s benchmaxxing. you gotta train on like some other stuff. but also the scores need to be super good</p>&mdash; will brown (@willccbb) <a href="https://twitter.com/willccbb/status/1993009122644836831?ref_src=twsrc%5Etfw">November 24, 2025</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Intelligence is not parameter count.**

    The existence of a simple equation with such powerful expressivity deomonstrates that model complexity cannot be determined by counting parameters alone. The one-parameter model exploits a often-overlooked fact: a single real-valued parameter can encode an unbounded amount of information by hiding complexity in its digits rather than in parameter count. Larger models should not be assumed to be strictly smarter. Instead, what parameter count actually measures is computational cost, not intelligence. FLOPs scale with parameters and precision, making parameter count a useful proxy for the limiting resources in ML: compute and memory.

    Of course we would never train a model by shoving all the information into one parameter. In practice, parameter counts usually assumes finite-precision weights (e.g. fp32), not infinite precision. That's why the one-parameter model is more of a thought experiment than a practical proposal.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Intelligence is compression.**

    To compress data, you must find regularities in it and finding regularities fundamentally requires intelligent pattern matching. If [intelligence is compression]((https://en.wikipedia.org/wiki/Hutter_Prize)), then our one-parameter model has all the intelligence of a phonebook. It achieves zero compression and is just a nice lookup table. It cannot discover patterns or extract structure. The one-parameter model simply stores the raw data and uses the precision $p$ as a tunable recovery knob.

    Real compression requires understanding. If you want to measure the complexity and expressivity of machine learning models, measure their compression. Use minimum description length or Kolmogorov complexity. These techniques capture whether a model has actually learned the underlying patterns. They cut through the illusion of parameter counts and reveal what the model truly understands.

    Prof. Albert Gu's paper [ARC-AGI without pretraining](https://iliao2345.github.io/blog_posts/arc_agi_without_pretraining/arc_agi_without_pretraining.html) actually does this right. They used a general-purpose compression algorithm to solve ARC-AGI without training on the test set. Our one-parameter model is a degenerate version of the same idea.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **The ARC-AGI Benchmark**

    ARC-AGI was intentionally designed to resist overfitting. It uses a private test set for official scoring, making training on test impossible. (Our one-parameter model only trained on the public eval set, not the private one.)

    Yet modern reasoning models may still be overfitting on ARC-AGI, just not in the traditional sense. Instead of training directly on the test set, reasoning models are clever enough to exploit distributional similarities between public and private splits, a meta-level form of overfitting to the benchmark's structural patterns. The ARC-AGI organizers [acknowledge](https://arcprize.org/blog/arc-prize-2025-results-analysis) this phenomenon, raising concerns about overfitting on their own benchmark.

    However, the fundamental problem runs deeper. Many ARC-AGI solutions appear benchmark-specific, using synthetic data and abstractions tailored to these visual-grid puzzles. How many of these solutions have inspired downstream improvements in LLMs or other modes of intelligence? ARC-AGI is a necessary but not sufficient condition for AGI. I hope these techniques prove to be good for more than just ARC-AGI's delightful puzzles, driving broader innovation in the field of AI.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Is this AGI?**
    """)
    return


@app.cell
def _():
    from marimo import Html

    tweet_html = """
    <blockquote class="twitter-tweet"><p lang="en" dir="ltr">Wait, this sounds incredible useful! Can we just have a model with 0 entropy, 0 hallucinations, that just acts like a retrieval database over its training dataset? Also sounds like a great way to solve the traceability problem. Why don&#39;t the AI labs just make something like that? <a href="https://t.co/n37BIqqQPO">pic.twitter.com/n37BIqqQPO</a></p>&mdash; Тsфdiиg (@tsoding) <a href="https://twitter.com/tsoding/status/2016641844755443774?ref_src=twsrc%5Etfw">January 28, 2026</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
    """
    Html(tweet_html)
    return


@app.cell
def _(mo):
    mo.md(r"""
    This tweet got ~500k views suggesting AI labs should buikd "a model with 0 entropy, 0 hallucinations, that just acts like a retrieval database over its training dataset".

    Our one-parameter model does exactly that! It has zero entropy, zero hallucinations, and perfect retrieval over the training data -- all while being continuous, differentiable, and contained in a single parameter.

    However, this isn't the path to AGI; I do not think AI labs should build this. The real challenge of intelligence isn't memorizing and retrieving training data like a retreival database. It’s about generalization, interpolating between the training points to predict new unseen points in the evaluation set.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Closing Thoughts**

    This one-parameter model is a ridiculous thought experiment taken seriously. It pushes overfitting to its absurd limit, and I hope leaves you rethinking generalization, overfitting, and how we can actually measure intelligence. Personally, this was a lot of fun to work on too.

    All the code for my experiments can be found here: [https://github.com/eitanturok/one-parameter-model](https://github.com/eitanturok/one-parameter-model).

    If you liked this or want to chat, [reach out](https://eitanturok.github.io/)! I always love talking with interesting folks!

    Lastly, thanks to all those who gave me helpful feedback on this post: [Gilad](https://x.com/giladturok), [Jacob](https://x.com/JacobianNeuro), [Isaac](https://x.com/LiaoIsaac91893), [spike](https://x.com/spikedoanz), and others.


    To cite this blog post
    ```md
    @online{Turok2025ARCAGI,
    	author = {Eitan Turok},
    	title = {A one-parameter model that gets 100% on ARC-AGI-2},
    	year = {2025},
    	url = {https://eitanturok.github.io/one-parameter-model/},
    }
    ```
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    # Appendix
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Appendix A: Other Uses of the One-Parameter Model
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    The one-parameter model can be applied to all sorts of datasets beyond ARC-AGI-2. For instance, we can encode animal shapes with different values of $\alpha$
    """)
    return


@app.cell
def _(mo):
    animals_image = mo.image(
        "public/images/animals.png",
        width=800,
        caption="Encode animals with different values of alpha. Figure 1 of 'Real numbers, data science and chaos: How to fit any dataset with a single parameter'.",
        style={"display": "block", "margin": "0 auto"}
    )
    animals_image
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can find an $\alpha$ that perfectly predicts the fluctuations of the S&P 500 for ~6 months with
    ```py
    alpha = 0.9186525008673170697061215177743819472103574383504939864690954692792184358812098296063847317394708021665491910117472119056871470143410398692872752461892785029829514157709738923288994766865216570536672099485574178884250989741343121
    ```
    """)
    return


@app.cell
def _(mo):
    stocks_image = mo.image(
        "public/images/s_and_p.png",
        width=800,
        caption="Predict the S&P 500 with 100% accuracy until mid Febuary 2019. From Figure 9 of 'Real numbers, data science and chaos: How to fit any dataset with a single parameter'.",
        style={"display": "block", "margin": "0 auto"}
    )
    stocks_image
    return


@app.cell
def _(mo):
    mo.md(r"""
    And we can even find values of $\alpha$ that generate parts of the famous [CIFAR-10](https://en.wikipedia.org/wiki/CIFAR-10) dataset
    """)
    return


@app.cell
def _(mo):
    cifar10_image = mo.image(
        "public/images/cifar_10.png",
        width=800,
        caption="Encode samples that look like they are from cifar-10. From Figure 3 of 'Real numbers, data science and chaos: How to fit any dataset with a single parameter'.",
        style={"display": "block", "margin": "0 auto"}
    )
    cifar10_image
    return


@app.cell
def _(mo):
    mo.md(r"""
    Indeed, the one parameter model is incredibly versatile, able to train on all sorts of test sets.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Appendix B: Some Technical Critiques
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Two quick technical notes to the critics.

    **For the complexity theorists**, yes, this is cheating. We've violated the fundamental assumption of bounded-precision arithmetic. Most complexity problems assume we operate on a machine with an $\omega$-bit word-size. However, my one-parameter model assumes we can operate on a machine with infinite bit word-size.

    **For the deep learning theorists**, of course our one-parameter model can memorize any dataset. Our decoder contains $\sin$ which has an [infinite VC dimension](https://cseweb.ucsd.edu/classes/fa12/cse291-b/vcnotes.pdf), i.e. an unbounded hypothesis class, and is therefore infinitely expressive. It can learn anything. What is interesting about the one-parameter model is that it offers a tangible construction, not merely a claim of existence, for learning any dataset.
    """)
    return


if __name__ == "__main__":
    app.run()
