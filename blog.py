import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import json
    from functools import partial
    from multiprocessing import Pool

    import gmpy2
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from gmpy2 import sin as sin_ap, mpfr as float_ap, asin as arcsin_ap, sqrt as sqrt_ap, const_pi as pi_ap # ap = arbitrary precision
    from matplotlib import colors

    from src.one_parameter_model.data import local_arc_agi, process_arc_agi
    from src.one_parameter_model.utils import MinMaxScaler, Timing, tqdm, VERBOSE
    return (
        MinMaxScaler,
        Pool,
        Timing,
        VERBOSE,
        colors,
        gmpy2,
        json,
        local_arc_agi,
        np,
        partial,
        plt,
        process_arc_agi,
        tqdm,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # The One-Parameter Model That Broke ARC-AGI

    > I built a one-parameter model that gets 100% on ARC-AGI-1, the million-dollar reasoning benchmark that stumps GPT-5. Using chaos theory and some deliberate cheating, I crammed every answer into a single 866,970-digit number.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Intro""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    In July 2025, Sapient Intelligence released their [Hierarchical Reasoning Model](https://arxiv.org/pdf/2506.21734v1) (HRM) and the world went crazy. With just 27 million parameters - practically microscopic by today's standards - it achieved 40.3% on [ARC-AGI-1](https://arcprize.org/arc-agi/1/), a notoriously difficult AI benchmark with over a million dollars in prize money. What made this remarkable wasn't just the score, but that HRM outperformed models 1000x larger.

    I wondered: is it possible to make an even smaller model?

    **So I built a one parameter model that scores 100% on ARC-AGI-1.**

    The model is:

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

    where $x_i$ is the $i\text{th}$ datapoint, $p$ is the manually set precision parameter, and $\alpha$ is the learned scalar. All you need to get 100% on ARC-AGI-1 is:
    """
    )
    return


@app.cell
def _(gmpy2, json, mo):
    with open(mo.notebook_dir() / "public/alpha/ARC-AGI-1.json", "r") as f: data = json.load(f)
    alpha = gmpy2.mpfr(*data['alpha'])
    p = data['precision']

    # only display the first 1,000 digits of a so we don't break marimo
    mo.md(f"```py\nalpha={str(alpha)[:10_000]}\np={p}\n```")
    return (alpha,)


@app.cell
def _(alpha):
    assert len(str(alpha)) == 866_970
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    This number is 866,970 digits long and is effectively god in box, right? One scalar value that cracks one of the most challenging AI benchmarks of our time. Plug any ARC-AGI example into this bad boy and watch our model perfectly predict the solution!

    Sounds pretty impressive, right?

    Well, here's the thing - **it's complete nonsense.**

    What I've done here is use some clever mathematics from chaos theory to encode all the answers into a single, impossibly dense parameter. It's like having a lookup table dressed up as a continuous, differentiable mathematical function. There is no learning or generalization. It is pure memorization with trigonometry and a few extra steps. Rather than a breakthrough in reasoning, it's a very sophisticated form of cheating.

    My hope is that this deliberately absurd approach exposes the flaws in equating parameter count with intelligence, highlighting the difference between memorization and true generalization. As we unravel the surprisingly rich mathematics underlying this one-parameter model, it opens up deeper discussions about ARC-AGI, the HRM breakthrough, and the broader question of how we should actually measure machine intelligence.

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
    Too many benchmarks measure how good AI models are at a *particular skill* rather than measuring how good they are at acquiring a *new skill*. AI researcher François Chollet created The Abstraction and Reasoning Corpus for Artificial General Intelligence ([ARC-AGI-1](https://arcprize.org/arc-agi/1/)) to fix this. ARC-AGI-1 measures how well AI models can *generalize* to unseen tasks. It consists of problems that are [trivial](https://arcprize.org/arc-agi/1/) for humans but challenging for machines. Recently Chollet and his team released ARC-AGI-2 and ARC-AGI-3 as harder follow ups to ARC-AGI-1. However here we will focus on ARC-AGI-1. Currently there is a $1,000,000+ prize-pool for progress on ARC-AGI, inspiring a host of new research directions.

    **What makes ARC-AGI-1 different from typical benchmarks?**

    Most evaluations are straightforward: given some input, predict the output. ARC-AGI-1, however, is more complicated. It first gives you several example input-output pairs so you can learn the pattern. Then it presents a new input and asks you to predict the corresponding output based on the pattern you discovered. This structure means that a single ARC-AGI-1 task consists of:

    * several example input-output pairs
    * a question input
    * a question output

    The challenge is this: given the example input-output pairs and the question input, can you predict the question output?

    **What does an ARC-AGI-1 task actually look like?**

    ARC-AGI-1 consists of visual grid-based reasoning problems. Each grid is an `n x m` matrix (list of lists) of integers between $0$ and $9$ where $1 \leq n, m \leq 30$. To display the grid, we simply choose a unique color for each integer. Let's look at an example:
    """
    )
    return


@app.cell
def _(colors, plt):
    # from https://www.kaggle.com/code/allegich/arc-agi-2025-visualization-all-1000-120-tasks

    # 0:black, 1:blue, 2:red, 3:green, 4:yellow, # 5:gray, 6:magenta, 7:orange, 8:sky, 9:brown
    cmap = colors.ListedColormap(['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00', '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=0, vmax=9)

    def plot_one(ax, i, task, example_or_question, input_or_output, w=0.8):
        key = f"{example_or_question}_{input_or_output}"
        input_matrix = task[key][i]

        # grid
        ax.imshow(input_matrix, cmap=cmap, norm=norm)
        ax.grid(True, which='both', color='lightgrey', linewidth=1.0)
        plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])
        ax.set_xticks([x-0.5 for x in range(1 + len(input_matrix[0]))])
        ax.set_yticks([x-0.5 for x in range(1 + len(input_matrix))])
        ax.grid(visible= True, which = 'both', color = '#666666', linewidth = w)
        ax.tick_params(axis='both', color='none', length=0)

        # subtitle
        ax.set_title(f'\n{example_or_question.capitalize()} {i} {input_or_output[:-1].capitalize()}', fontsize=12, color = '#dddddd')

        # status text positioned at top right
        if example_or_question == 'question' and input_or_output == 'outputs':
            ax.text(1, 1.15, '? PREDICT', transform=ax.transAxes, ha='right', va='bottom', fontsize=10, fontweight='bold', color='#FF4136')
        else:
            ax.text(1, 1.15, '✓ GIVEN', transform=ax.transAxes, ha='right', va='bottom', fontsize=10, fontweight='bold', color='#2ECC40')


    def display_task(ds, split, i, size=2.5, w1=0.9):
        task = ds[split][i]
        n_examples = len(task['example_inputs'])
        n_questions  = len(task['question_inputs'])
        task_id = task["id"]

        wn=n_examples+n_questions
        fig, axs  = plt.subplots(2, wn, figsize=(size*wn,2*size))
        plt.suptitle(f'ARC-AGI-1 {split.capitalize()} Task #{i} (id={task_id})', fontsize=16, fontweight='bold', y=1, color = '#eeeeee')

        # plot train
        for j in range(n_examples):
            plot_one(axs[0, j], j, task, 'example', 'inputs',  w=w1)
            plot_one(axs[1, j], j, task, 'example', 'outputs', w=w1)

        # plot test
        for k in range(n_questions):
            plot_one(axs[0, j+k+1], k, task, 'question', 'inputs', w=w1)
            plot_one(axs[1, j+k+1], k, task, 'question', 'outputs', w=w1)

        axs[1, j+1].set_xticklabels([])
        axs[1, j+1].set_yticklabels([])
        axs[1, j+1] = plt.figure(1).add_subplot(111)
        axs[1, j+1].set_xlim([0, wn])

        # plot separators
        # for m in range(1, wn): axs[1, j+1].plot([m,m],[0,1],'--', linewidth=1, color='white')
        axs[1, j+1].plot([n_examples, n_examples], [0,1], '-', linewidth=5, color='white')

        axs[1, j+1].axis("off")

        # Frame and background
        fig.patch.set_linewidth(5) #widthframe
        fig.patch.set_edgecolor('black') #colorframe
        fig.patch.set_facecolor('#444444') #background

        plt.tight_layout(h_pad=3.0)
        # plt.show()
        return fig
    return (display_task,)


@app.cell
def _(local_arc_agi):
    ds = local_arc_agi("public/data/ARC-AGI-1")
    return (ds,)


@app.cell
def _(display_task, ds):
    display_task(ds, "train", 1)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Here, we see several grids, each with a bunch of colored cells. Most cells are black (0), some are green (3), and some are yellow (4). Each column shows an input-output pair.

    The first five columns are example input-output pairs that demonstrate the pattern. The sixth column, separated by the solid white line, is the actual question: given this new input, what should the output be?

    The color coding makes this clear. The green '✓ Given' means the model has access to this information while the red '? Predict' is the ground truth solution and is what the model must predict on its own. The example input-output pairs and the question input are green while the question output is red. We're showing the question output here (red) so you can see what the correct answer is, but during evaluation, the question output is completely hidden from the model.

    **Now, how do you solve this specific task?**

    Looking at the examples, the pattern here is clear: add yellow squares inside the enclosed green shapes. Yellow only appears in the "interior" of closed green boundaries. If the green cells don't form a complete enclosure, no yellow is added.

    Looking at the question input, we have a complicated looking shape, a green line that snakes around. But if you look closely, you can count that the input shape has 8 different encolosed shapes that need to be filled in with yellow squares. So in the output, we fill in all 8 "interior" regions with yellow squares.

    Looking at the question output, we can verify that this solution is indeed correct.

    Another task:
    """
    )
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

    Looking at the question input, we can follow this pattern. The question input contains a distinctive green frame that contrasts  with the surrounding black, blue, and red cells. Therefore we should output everything inside the green frame.

    Looking at the question output, we see that this is indeed the correct answer.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""There are hundreds of tasks like this in ARC-AGI-1. Solving each task requires deducing new patterns and generalizing to unforeseen tasks, something it is quite hard for the current crop of AI models.""")
    return


@app.cell
def _(mo):
    arc_agi_1_leaderboard_image = mo.image(
        mo.notebook_dir() / "public/images/arc-agi-1-leaderboard.png",
        width=800,
        caption="Performance on private eval set of ARC-AGI-1.",
        style={"display": "block", "margin": "0 auto"}
    )
    arc_agi_1_leaderboard_image
    return


@app.cell
def _(mo):
    mo.md(f"""Even the world's best models struggle on ARC-AGI-1, often scoring under $50\%$. `o3-preview (Low)` has the highest score of $75.7\%$ but costs a staggering $\$200$ per task. GPT-5 (High) is much more efficient, scoring $65.7\%$ with a cost of only $\$0.51$ per task. However, many other frontier models -- Claude, Gemini-2.5, and Deepseek -- struggle to even get half of the questions right. In contrast, humans [score](https://arcprize.org/leaderboard) get $98\%$ of questions right. That's why there exists a $\$1,000,000$ [competition](https://arcprize.org/competitions/2025/) to open source a solution to ARC-AGI-1, ARC-AGI-2, and ARC-AGI-3. It's that difficult. (Note: when the HRM paper was released the scores were *much* lower; there has been a lot of recent progres on ARC-AGI-1.)""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # The HRM Drama

    > \> see "miraculous" 27M model beats frontier AI
    >
    > \> methodology: train on eval set with learnable task tokens
    >
    > \>call it "reasoning"
    >
    > \> profit -- [twitter user](https://x.com/b_arbaretier/status/1951701328754852020)
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""The recent HRM is a fascinating model, inspired by the human brain with "slow" and "fast" loops of computation. It gained a lot of attention for it's amazing performance on ARC-AGI-1 despite its tiny size of 27M parameters.""")
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
    HRM scored 40.3% on ARC-AGI-1 while SOTA models like o3-mini-high and Claude-3.7-8k scored 34.5%, and 21.2% respectively. It beat Anthropic's best model by nearly ~2x! Similarly, it outperformed o3-mini-high and Claude-3.7-8k on ARC-AGI-2, but be warned that the ARC-AGI-2 the scores are so low that they are more much suspectable to noise.

    The results almost seemed to be too good to be true. How can a tiny 27M parameter model from a small lab be crushing some of the world's best models, at a fraction of their size?

    Turns out, HRM "trained on test"
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
    In their paper, the HRM authors admitted to training on the public eval set of ARC-AGI-1! On github, the HRM authors clarified that they only trained on the *examples* of the public eval set, not the *questions* of the public eval set. This "contraversy" set AI twitter on fire [[1](https://x.com/Dorialexander/status/1951954826545238181), [2](https://github.com/sapientinc/HRM/issues/18), [3](https://github.com/sapientinc/HRM/issues/1) [4](https://github.com/sapientinc/HRM/pull/22) [5](https://x.com/b_arbaretier/status/1951701328754852020)] ! Does this actually count as "training on test"? On one hand, you can never train on the data used to measure model perfomance. On the other hand, they never actually trained on the the questions used to measure model performance, just the examples associated with them.

    **What exactly is the difference between training on *examples* VS *questions* in ARC-AGI-1?**

    Consider a task from the public *eval* set of ARC-AGI-1:
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
    This task has two example input-output pairs and a question input-output pair. Training on just the *examples* means that HRM was trained only on the two examples, left of the vertical white line. It was *not* trained on the question, to the right of the vertical white line. To measure model performance, the model was then evaluated on the *questions*. This means that the model saw the examples, which are in the same distribution as the question, but never the actual questions themselves.

    So does this count as "data leakage" or "cheating"? While the internet still debates, the ARC-AGI organizers ultimately accepted the HRM submission so I guess they decided it is okay to train the *examples* of the public eval set. At the end of this episode, one comment by HRM's lead author caught my attention:
    > "If there were genuine 100% data leakage - then model should have very close to 100% performance (perfect memorization)." -   [Guan Wang](https://github.com/sapientinc/HRM/issues/1#issuecomment-3113214308)

    Well, that got me curious. What would happen if we really did memorize everything?

    Is it possible to get 100% on ARC-AGI-1 with full data leakage? If we train on the questions of the public eval set, not just the examples, could we beat HRM's 40.3% on ARC-AGI-1? Could we still do it with very few parameters, like HRM? How far can we push this?
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
    My goal was simple: create the tiniest possible model that achieves perfect performance on ARC-AGI-1 by blatantly training on the public eval set, both the examples and questions. We would deviate from HRM's acceptable approach -- training on just the examples of the public eval set -- and enter the morally dubious definetly-cheating territory -- training on the examples *and questions* of the public eval set.

    Now, the obvious approach would be to build a dictionary - just map each input directly to its corresponding output. But that's boring and lookup tables aren't nice mathematical functions. They're discrete, discontinuous, and definitely not differentiable. We need something else, something more elegant and interesting. To do that, we are going to take a brief detour into the world of chaos theory.

    *Before diving in, I need to acknowledge that this technique comes from one of my all-time favorite papers: [Real numbers, data science and chaos: How to fit any dataset with a single parameter](https://arxiv.org/abs/1904.12320) by [Laurent Boué](https://www.linkedin.com/in/laurent-bou%C3%A9-b7923853/?originalSubdomain=il). This paper is really a gem, a top ten paper of all time due its sheer creativity. Boué's paper, in turn, was originally inspired by [Steven Piantadosi](https://colala.berkeley.edu/people/piantadosi/)'s research [One parameter is always enough](https://colala.berkeley.edu/papers/piantadosi2018one.pdf). Let's dive in.*

    The dyadic map $\mathcal{D}$ is a simple one-dimensional chaotic system defined as

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

    mean we apply the dyadic map $k$ times to $a$. What does the orbit $(a, \mathcal{D}^1(a), \mathcal{D}^2(a), ...)$ look like?

    * If $a = 0.5$, the orbit is $(0.5, 0.0, 0.0, 0.0, 0.0, 0.0, ...)$.
    * If $a = 1/3$, the orbit is $(0.333, 0.667, 0.333, 0.667, 0.333, 0.667, ..., )$
    * If $a = 0.43085467085$, the orbit is $(0.431, 0.862, 0.723, 0.447, 0.894, 0.787, ...)$

    One orbit seems to end in all zeros, another bounces back and forth between $0.333$ and $0.667$, and a third seems to have no pattern at all. On the surface, these orbits do not have much in common. But if we take a closer look, they all share the same underlying pattern.

    Let's revisit the third orbit for $a = 0.43085467085$:

    $$
    (a, \mathcal{D}^1(a), \mathcal{D}^2(a), \mathcal{D}^3(a), \mathcal{D}^4(a), \mathcal{D}^5(a), ...)
    =
    (0.431, 0.862, 0.723, 0.447, 0.894, 0.787, ...)
    $$

    but this time we will analyze its binary representation:

    | Iterations | Decimal | Binary | Observation |
    |------------|------------------------|----------------------|-------------|
    | 0 | $a\phantom{^4(a)} = 0.431$ | $\text{bin}(a)\phantom{^4(a)} = 0.011011...$ | Original number |
    | 1 | $D^1(a) = 0.862$ | $\text{bin}(D^1(a)) = 0.11011...$ | First bit of $a$ $(0)$ removed |
    | 2 | $D^2(a) = 0.723$ | $\text{bin}(D^2(a)) = 0.1011...$ | First two bits of $a$ $(01)$ removed |
    | 3 | $D^3(a) = 0.447$ | $\text{bin}(D^3(a)) = 0.011...$ | First three bits of $a$ $(011)$ removed |
    | 4 | $D^4(a) = 0.894$ | $\text{bin}(D^4(a)) = 0.11...$ | First four bits of $a$ $(0110)$ removed |
    | 5 | $D^4(a) = 0.787$ | $\text{bin}(D^5(a)) = 0.1...$ | First four bits of $a$ $(01101)$ removed |

    Looking at the Binary column, we see that **every time we apply the dyadic map, the most significant bit is removed**! We start off with $0.011011$, and then applying $\mathcal{D}$ once removes the leftmost $0$ to get $0.11011$, and applying $\mathcal{D}$ another time removes the leftmost $1$ to get $0.1011$. Although the orbit appears irregular in its decimal representation, a clear pattern emerges from the binary representation.

    What is going on here?

    Each time we call $D(a) = (2a) \mod 1$, we double and truncate $a$. The doubling shifts every binary digit one place to the left and the truncation throws away whatever digit lands in the one's place. In other words, each application of $\mathcal{D}$ peels off the first binary digit and throws it away. **If we apply the dyadic map $k$ times, we remove the first $k$ bits of $a$.**

    We can see this process also holds for our other orbits:

    * If $a = 0.5$, we get the orbit $(0.5, 0.0, 0.0, 0.0, 0.0, 0.0, ...)$ because $\text{bin}(a) = 0.100000...$ and after discarding the first bit, which is a $1$, we are left with all zeros.
    * If $a = 1/3$, we get the orbit $(0.333, 0.667, 0.333, 0.667, 0.333, 0.667, ...)$ because $\text{bin}(a) = 0.010101...$, an infinite sequence of bits alternating between $1$ and $0$. When the bits start with a 0, we get $0.010101...$ which is $1/3 = 0.333$ in decimal. And when the bits start with a $1$, $0.10101...$, we get $2/3 = 0.667$ in decimal.

    Remarkably, these orbits are all governed by the same rule: remove one bit of information every time the dyadic map is applied. As each application of $\mathcal{D}$ removes another bit, this moves us deeper into the less significant digits of our original number -- the digits that are most sensitive to noise and measurement errors. A tiny change in $a$​, affecting the least significant bits of $a$, would eventually bubble up to the surface and completely change the orbit. That's why this system is so chaotic -- it is incredibly sensitive to even the smallest changes in the initial value $a$.

    (Note: we always compute the dyadic map on *decimal* numbers, not binary numbers; however, conceptually it is helpful to think about the binary representations of the orbit.)
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # The Dyadic Map As An ML Model
    > "I'm going to pretend to be an ML Model, now" - the Dyadic Map
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    We've discovered something remarkable: each application of $\mathcal{D}$ peels away exactly one bit. But here's the question: if the dyadic map can systematically extract a number's bits, is it possible to put information in those bits in the first place? **What if we encode our dataset into a number's bits (`model.fit`) and then use the dyadic map as the core of a predictive model, extracting out the answer bit by bit (`model.predict`)?**

    Suppose our dataset contains the three numbers we saw before

    $$
    \mathcal{X}
    =
    \{x_0, x_1, x_2\}
    =
    \{0.5, 1/3,  0.43085467085\}.
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

    The number $\alpha$ is carefully engineered so that it is a decimal number whose bits contain our entire dataset's binary representation. That's right: **we've just compressed our entire dataset into a single scalar decimal number!**

    But here's the question: how do we get our data back out? This is where the dyadic map becomes our extraction tool.

    Trivially, we know the first 6 bits of $\alpha$ contains $b_0$. So we'll just record the first $6$ bits to get $b_0$.

    | Iterations | Decimal | Binary | First $6$ bits |
    |------------|------------------------|----------------------|-------------|
    | 0 | $\alpha = 0.50522994995117188$ | $\text{bin}(\alpha) = 0.\underbrace{100000}_{b_0}\underbrace{010101}_{b_1}\underbrace{011011}_{b_2}$ | $b_0$ |

    To get $b_1$, remember that each application of $\mathcal{D}$ strips away the leftmost binary digit. So $D^6(\alpha)$ strips away the first $6$ bits of $\alpha$, which just removes $b_0$, and leaves us with $b_1, b_2$. We'll then record the first $6$ bits to get $b_1$.

    | Iterations | Decimal | Binary | First $6$ bits |
    |------------|------------------------|----------------------|-------------|
    | 0 | $\alpha = 0.50522994995117188$ | $\text{bin}(\alpha) = 0.\underbrace{100000}_{b_0}\underbrace{010101}_{b_1}\underbrace{011011}_{b_2}$ | $b_0$ |
    | 1 | $\mathcal{D}^1(\alpha) = ....$ | $\text{bin}(D^6(\alpha)) = 0.\underbrace{010101}_{b_1}\underbrace{011011}_{b_2}$ | $b_1$|

    To get $b_2$, apply $\mathcal{D}$ another 6 times, $\mathcal{D}^{12}(\alpha)$, removing another 6 bits of $\alpha$, i.e. $b_1$, and leaving us with just $b_2$. We'll then record the first $6$ bits to get $b_2$.

    | Iterations | Decimal | Binary | First $6$ bits |
    |------------|------------------------|----------------------|-------------|
    | 0 | $\alpha = 0.50522994995117188$ | $\text{bin}(\alpha) = 0.\underbrace{100000}_{b_0}\underbrace{010101}_{b_1}\underbrace{011011}_{b_2}$ | $b_0$ |
    | 1 | $\mathcal{D}^1(\alpha) = ....$ | $\text{bin}(D^6(\alpha)) = 0.\underbrace{010101}_{b_1}\underbrace{011011}_{b_2}$ | $b_1$|
    | 2 | $\mathcal{D}^2(\alpha) = ....$ | $\text{bin}(D^{12}(\alpha)) = 0.\underbrace{011011}_{b_2}$ | $b_1$|

    Using the dyadic map as a data extraction machine, we've just recovered the original $6$-bit representation of our data $\mathcal{B} = \{b_0, b_1, b_2 \}$. If we convert these back to decimal, we'll recover our original data

    $$
    \tilde{\mathcal{X}}
    =
    \{
    \tilde{x}_1
    \tilde{x}_2
    \tilde{x}_3
    \}
    =
    \{
        3, 4, 5
    \}
    $$

    where $\tilde{x}_i = \text{dec}(b_i)$ is a function that converts binary to decimal. Notice, that $\tilde{x}_3 \neq x_3$ because we only saved the first $6$ bits of $x_3$ and not the entire thing. But this is a pretty great approximation!

    Think about what we've accomplished here. We just showed that you can take a dataset compress it down to a single real number, $\alpha$. Then, using nothing more than repeated doubling and truncation via $\mathcal{D}$, we can perfectly recover every data point in binary $b_0, b_1, b_2$. The chaotic dynamics of the dyadic map, which seemed like a nuisance, turns out to be the precise mechanism we need to systematically access that information.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    The algorithm itself is deceptively simple once you see the pattern:

    > **Encoding Algorithm:**
    > Given a dataset $\mathcal{X} = \{x_0, ..., x_n\}$ where $x_i \in [0, 1]$, encode the dataset into $a$:
    >
    > 1. Convert each number to binary with $p$ bits of precision $b_i = \text{bin}_p(x_i)$ for $i=1, ..., n$
    > 2. Concatenate into a single binary string $b = b_0 \oplus  ... \oplus b_n$
    > 3. Convert to decimal $\alpha = \text{dec}(b)$


    The result is a single, decimal, scalar number $\alpha$ with $np$ bits of precision that contains our entire dataset. We can now discard $\mathcal{X}$ entirely.

    > **Decoding Algorithm:**
    > Given sample index $i$ and the encoded number $\alpha$, recover sample $\tilde{x_i}$:
    >
    > 1. Apply the dyadic map $\mathcal{D}$ exactly $ip$ times $\tilde{x}'_i = \mathcal{D}^{ip}(\alpha) = (2^{ip} \alpha) \mod 1$
    > 2. Extract the first $p$ bits of $\tilde{x}'_i$'s binary representation $b_i = \text{bin}_p(\tilde{x}'_i)$
    > 3. Covert to decimal $\tilde{x}_i = \text{dec}(b_i)$

    The precision parameter $p$ controls the trade-off between accuracy and storage efficiency. Since $\tilde{x}_i = \text{dec}(b_i) = \text{dec}(\text{bin}_p(x_i))$, we have $\tilde{x}_i \approx x_i$ with error bound $2^{-p}$. What makes this profound is the realization that we're not really "storing" information in any conventional sense. We're encoding it directly into the bits of a real number, exploiting it's infinite precision, and then using the dyadic map to navigate through that number and extract exactly what we need, when we need it.

    Mathematically, we can express this with two functions the encoder $g: [0, 1]^n \to [0, 1]$ that compresses the dataset and the decoder $f: \overbrace{[0, 1]}^{\alpha} \times \overbrace{\mathbb{Z}_+}^{p} \times \overbrace{[n]}^i \to [0, 1]$ that extracts individual data points:

    $$
    \begin{align*}
    \alpha
    &=
    g(p, \mathcal{X}) := \text{dec} \Big( \bigoplus_{x \in \mathcal{X}} \text{bin}_p(x) \Big)
    \tag{4}
    \\
    \tilde{x}_i
    &=
    f_{\alpha, p}(i) := \text{dec} \Big( \text{bin}_p \Big( \mathcal{D}^{ip}(\alpha) \Big) \Big)
    \end{align*}
    $$

    where $\oplus$ means concatenation.

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

    to that beautiful decoder function I promised you at the start of the blog

    $$
    f_{\alpha, p}(x)
    =
    \sin^2 \Big(
        2^{i p} \arcsin^2(\sqrt{\alpha})
    \Big)
    ?
    $$

    In this section we will "apply makeup" to the first function to get something that looks more like the second function. We will keep the core logic the same but make the function more ascetically pleasing. To do this, we will need another one-dimensional chaotic system, the [logistic map](https://en.wikipedia.org/wiki/Logistic_map) at $r=4$ on the unit interval:

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


@app.cell
def _(np, plt):
    def _():
        def L(a): return 4 * a * (1 - a)
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


@app.cell
def _(mo):
    mo.md(
        r"""
    What does the logistic orbit $(a_L, \mathcal{L}^1(a_L), \mathcal{L}^2(a_L), \mathcal{L}^3(a_L), ...)$ look like? Similar or different to the dyadic orbit $(a_D, \mathcal{D}^1(a_D), \mathcal{D}^2(a_D), \mathcal{D}^3(a_D), ...)$?

    * If $a_L = a_D = 0.5$, the logistic orbit is $()$ and the dyadic orbit is $(0.5, 0.0, 0.0, 0.0, 0.0, 0.0, ...)$.
    * If $a_L = 1/3$, the logistic orbit is $()$ and the dyadic orbit is $(0.333, 0.667, 0.333, 0.667, 0.333, 0.667, ..., )$
    * If $a_L = 0.43085467085$, the logistic orbit is $()$ and the dyadic orbit is $(0.431, 0.862, 0.723, 0.447, 0.894, 0.787, ...)$

    These orbits look nothing alike!

    However, [Topological conjugacy](https://en.wikipedia.org/wiki/Topological_conjugacy) tells us these two maps are *actually* the same. Not similar. Not analgous. The same. They have identical orbits, the exact same chaotic trajectories, simply expressed in different coordinates. The logistic map, for all its smooth curves and elegant form, is actually doing discrete binary operations under the hood, just like the dyadic map (and vice versa). Formally, two functions are topologically conjugate if there exists a homeomorphism, fancy talk for a change of coordinates, that perfectly takes you from one map to another. The change of coordinates here is

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
    The function $\phi$ has a period of 1, meaning it repeats the same values every time $a_D$ increases by $1$. This periodicity is crucial because it allows us to drop the modulo operation when transforming from the dyadic space to the logistic space:

    $$
    \begin{align*}
    \phi(a_D \mod 1) = \phi(a_D)
    \tag{9}
    \end{align*}
    $$

    To go back and forth between the dyadic and logistic maps, we do

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

    where $\phi$ takes us to the logistic space and operates on the output and $\phi^{-1}$ takes us to the dyadic space and operates on the input.

    This is astonishing! $\phi$ is just a sin wave squared and with a period of one. It's inverse, $\phi^{-1}$ is even weirder looking with an $\arcsin$. But somehow these functions allow us to bridge the two maps $\mathcal{D}$ and $\mathcal{L}$!

    Moreover, $\phi$ and $\phi^{-1}$ perfectly relate *every* single point in the infinite orbits of $\mathcal{D}$ and $\mathcal{L}$:

    $$
    (a_D, \mathcal{D}^1(a_D), \mathcal{D}^2(a_D), ...) = (a_L, \mathcal{L}^1(\phi^{-1})(a_L), \mathcal{L}^2(\phi^{-1})(a_L), ...)
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
    This means the dyadic and logistic orbit for $a_D = a_L = 0.43085467085$

    is

    * If $a_L = 0.5$, the logistic orbit is $()$ and the dyadic orbit is $(0.5, 0.0, 0.0, 0.0, 0.0, 0.0, ...)$.
    * If $a_L = 1/3$, the logistic orbit is $()$ and the dyadic orbit is $(0.333, 0.667, 0.333, 0.667, 0.333, 0.667, ..., )$
    * If $a_L = 0.43085467085$, the logistic orbit is $()$ and the dyadic orbit is $(0.431, 0.862, 0.723, 0.447, 0.894, 0.787, ...)$

    are related by
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    **Can we use the topological conjugacy of $\mathcal{D}$ and $\mathcal{L}$ as makeup?**

    While $\mathcal{D}$ is ugly and discontinuous, $\mathcal{L}$ is smooth and differentiable. We can use the logistic map as "makeup" to hide the crude dyadic operations. We want our decoder to use $\mathcal{L}$ instead of $\mathcal{D}$. But for the encoder to glue together the bits of our dataset, we needs to be in the dyadic space so we can do our clever bit manipulation. Here's the strategy:

    1. Encoder: Work in dyadic space (where bit manipulation works) (use $\phi$) but output parameter in logistic space (use $\phi^{-1}$)
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

    Again, the precision parameter $p$ controls the trade-off between accuracy and storage efficiency. Since $\tilde{x}_i = \text{dec}(b_i) = \text{dec}(\text{bin}_p(x_i))$ only contains the first $p$ bits of $x_i$ and discards the remaining bits, we have $\tilde{x}_i \approx x_i$ with error bound $\frac{\pi}{2^{p-1}}$.
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

    where $\oplus$ means concatenation. The decoder here is quite close to the function I promoised at the start of the blog

    $$
    f_{\alpha, p}(x)
    =
    \sin^2 \Big(
        2^{x p} \arcsin^2(\sqrt{\alpha})
    \Big)
    $$

    but is wrapped with an extra $\text{dec}$ and $\text{bin}_p$. We are so close.

    Doesn't this look beautiful? No modulo operations, no explicit bit extraction, no discontinuous jumps. The makeup looks great! And the decoder uses the logistic map $\mathcal{L}$ to give us the exact equation we got at the beginning of the blog!
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    In practice, our parameter $\alpha$ contains $np$ bits of precision, far exceeding the $32$ or $64$ bits that standard computers can handle. So we use arbitrary precision arithmetic libraries like [gmpy2](https://github.com/aleaxit/gmpy), which can perform computation with any level of precision we specify.

    This capability allows us to simplify the decoder equation. We can initially set our working precision to $np$ bits when computing $\mathcal{D}^{ip}(\alpha)$. Then we can simply change the working precision to $p$ bits and `gmpy2` will automatically look at the first $p$ bits of $\mathcal{D}^{ip}(\alpha)$. With `gmpy2`, there is no need to explicitally convert $\tilde{x}'_i = \mathcal{D}^{ip}(\alpha)$ to binary, extract the first $p$ bits, and then convert it back to decimal -- the library will take care of this for us. We can therefore skip the last two steps of the decoder algorithm because $\tilde{x}'_i = \tilde{x}_i$ when using `gmpy2`. The decoder simplifies then to:

    $$
    \begin{align*}
    \tilde{x}_i
    &=
    f_{\alpha,p}(i) := \mathcal{D}^{ip}(\alpha)
    \tag{5}
    \end{align*}
    $$
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
    Finally, what we've been waiting for. The code implementation!

    First, let's first define some basic helper functions for $\text{bin}_p(), \text{dec}(), \phi(), \phi^{-1}()$. Note that we compute $\phi^{-1}$ in numpy but use our aribtrary precision library to compute $\phi$.
    """
    )
    return


@app.cell
def _(gmpy2, mo, np):
    def decimal_to_binary(y_decimal, precision):
        if not isinstance(y_decimal, np.ndarray): y_decimal = np.array(y_decimal)
        if y_decimal.ndim == 0: y_decimal = np.expand_dims(y_decimal, 0)

        powers = 2**np.arange(precision)
        y_powers = y_decimal[:, np.newaxis] * powers[np.newaxis, :]
        y_fractional = y_powers % 1 # extract the fractional part of y_powers
        binary_digits = (y_fractional >= 0.5).astype(int).astype('<U1')
        return np.apply_along_axis(''.join, axis=1, arr=binary_digits).tolist()

    def binary_to_decimal(y_binary):
        # indicate the binary number is a float in [0, 1], not an int
        fractional_binary = "0." + y_binary
        return gmpy2.mpfr(fractional_binary, base=2)

    def phi_inverse(x): return np.arcsin(np.sqrt(x)) / (2.0 * np.pi)

    def phi(x): return gmpy2.sin(x * 2 * gmpy2.const_pi()) ** 2

    mo.show_code()
    return binary_to_decimal, decimal_to_binary, phi, phi_inverse


@app.cell
def _(mo):
    mo.md(
        r"""
    Next, let's implement the logistic map

    $$
    \mathcal{L}^{ip}(\alpha) = \sin^2 \Big(2^{ip} \arcsin(\sqrt{\alpha}) \Big)
    $$

    with aribtrary precision from `gmpy2`. We set the precision to $np$ bits with `total_prec`, compute $\mathcal{L}^{ip}(\alpha)$, and then truncate to first `p` bits before casting to a regular python float. We then parallelize the entire function to make it even faster.
    """
    )
    return


@app.cell
def _(Pool, gmpy2, mo, np, partial, tqdm):
    def logistic_decoder_single(total_prec, alpha, prec, idx):
        gmpy2.get_context().precision = total_prec
        val = gmpy2.sin(gmpy2.mpfr(2) ** (idx * prec) * gmpy2.asin(gmpy2.sqrt(alpha))) ** 2
        return float(gmpy2.mpfr(val, precision=prec))

    def logistic_decoder(total_prec, alpha, prec, idxs, workers):
        # compute sequentially if workers==0, otherwise compute in parallell
        if workers == 0:
            return np.array([logistic_decoder_single(total_prec, alpha, prec, idx) for idx in tqdm(idxs)])
        fxn = partial(logistic_decoder_single, total_prec, alpha, prec)
        with Pool(workers) as p:
            return np.array(list(tqdm(p.imap(fxn, idxs), total=len(idxs), desc="Logistic Decoder")))

    mo.show_code()
    return (logistic_decoder,)


@app.cell
def _(mo):
    mo.md(r"""Finally, we can define our one parameter model.""")
    return


@app.cell
def _(
    MinMaxScaler,
    Timing,
    VERBOSE,
    binary_to_decimal,
    decimal_to_binary,
    gmpy2,
    logistic_decoder,
    mo,
    np,
    phi,
    phi_inverse,
):
    class ScalarModel:
        def __init__(self, precision, workers=0):
            self.precision = precision # binary precision, not decimal precision, for a single number
            self.workers = workers

        @Timing("fit: ", enabled=VERBOSE)
        def fit(self, X, y):
            # if the dataset is unsupervised, treat the data X like the labels y
            if y is None: y = X

            self.y_shape = y.shape[1:] # pylint: disable=attribute-defined-outside-init
            self.total_precision = y.size * self.precision # pylint: disable=attribute-defined-outside-init

            # scale labels to be in [0, 1]
            self.scaler = MinMaxScaler() # pylint: disable=attribute-defined-outside-init
            y_scaled = self.scaler.scale(y.flatten())

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

        @Timing("predict: ", enabled=VERBOSE)
        def predict(self, idxs):
            y_size = np.array(self.y_shape).prod()
            full_idxs = (np.tile(np.arange(y_size), (len(idxs), 1)) + idxs[:, None] * y_size).flatten()
            raw_pred = logistic_decoder(self.total_precision, self.alpha, self.precision, full_idxs, self.workers)
            return self.scaler.unscale(raw_pred).reshape((-1, *self.y_shape))

        def fit_predict(self, X, y): return self.fit(X, y).predict(X)

    mo.show_code()
    return (ScalarModel,)


@app.cell
def _(mo):
    mo.md(
        r"""
    /// details | Can you walk me through what `ScalarModel` does line-by-line?


    Let's walk through this big chunk of code line-by-line.

    We initialize the model with the desired precision $p$ and the number of workers for running our decoder in parallel.
    ```py
    class ScalarModel:
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
    Let's try it our model out on ARC-AGI-1!

    We will use the public eval set from ARC-AGI-1 which has 400 tasks. We can ignore the example input-output pairs and only look at the question inputs-output pairs because we are only actually predictioning the question outputs given the question inputs.
    """
    )
    return


@app.cell
def _(ds, mo, process_arc_agi):
    X, y = process_arc_agi(ds)
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
    mo.md(r"""Now we are ready to "train" our model and compress our entire dataset into $\alpha$. It is super easy.""")
    return


@app.cell
def _(ScalarModel, X, mo, y):
    precision = 8
    model = ScalarModel(precision)
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
    y_pred = model.predict(np.array([1]))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
