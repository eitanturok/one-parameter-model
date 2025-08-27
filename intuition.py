import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import altair as alt
    import pandas as pd
    import numpy as np
    return alt, np, pd


@app.cell
def _(np):
    def decimal_to_binary(y_decimal, precision):
        if not isinstance(y_decimal, np.ndarray): y_decimal = np.array(y_decimal)
        if y_decimal.ndim == 0: y_decimal = np.expand_dims(y_decimal, 0)
    
        powers = 2**np.arange(precision)
        y_powers = y_decimal[:, np.newaxis] * powers[np.newaxis, :]
        y_fractional = y_powers % 1 # extract the fractional part of y_powers
        binary_digits = (y_fractional >= 0.5).astype(int).astype('<U1')
        return np.apply_along_axis(''.join, axis=1, arr=binary_digits).tolist()[0]
    return (decimal_to_binary,)


@app.cell
def _(alt, pd):
    def add_cell(text, colors, top_text, border_colors, new_text, new_color="white", new_top_text="", new_border_color="white", left=True):
        idx = 0 if left else len(text)
        text.insert(idx, new_text)
        colors.insert(idx, new_color)
        top_text.insert(idx, new_top_text)
        border_colors.insert(idx, new_border_color)
        return text, colors, top_text, border_colors

    def make_chart(df, rect_width):
        # Create rectangles
        rects = alt.Chart(df).mark_rect(
            strokeWidth=2, width=rect_width, height=40
        ).encode(
            x=alt.X('x:O', axis=None), 
            y=alt.Y('y:O', axis=None), 
            color=alt.Color('color:N', scale=None),
            stroke=alt.Color('border:N', scale=None)
        )

        labels = alt.Chart(df).mark_text(fontSize=30).encode(x='x:O', y='y:O', text='text:N')
        top_labels = alt.Chart(df).mark_text(fontSize=16, dy=-33).encode(x='x:O', y='y:O', text='top_text:N')
        
        return (rects + labels + top_labels).properties(width=len(df) * (rect_width + 4), height=90)

    def display_digits(
        num_cells=8, colors=None, text=None, top_text=None, show_ellipses=True,
        var_symbol="a", rect_width=30, show_decimal=True
    ):
        border_colors = ['black'] * num_cells

        if show_decimal:
            text, colors, top_text, border_colors = add_cell(text, colors, top_text, border_colors, "0.")
        if var_symbol:
            text, colors, top_text, border_colors = add_cell(text, colors, top_text, border_colors, var_symbol+"=")
        if show_ellipses:
            text, colors, top_text, border_colors = add_cell(text, colors, top_text, border_colors, "...", left=False)
    
        x = list(range(len(text)))
        df = pd.DataFrame({'x': x, 'y': 0, 'color': colors, 'text': text, 'border': border_colors, 'top_text': top_text})
        return make_chart(df, rect_width)

    def digits(num, precision=None, base=None, color="white", var_symbol="a"):
        num_cells = 8 if precision is None else precision
        colors = [color] * num_cells
        base = 2 if isinstance(num, str) else 10
        if base == 10:
            assert num < 1, f'Expected num<1 but got {num=}'
            num = str(num).lstrip('0.')
        text = ([digit for digit in num]+ ['0'] * max(0, num_cells - len(num or [])))[:num_cells]

        subscript_map = {'0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄', '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉'}
        top_text = [("b" if base == 2 else "a") + ''.join([subscript_map[digit] for digit in str(i+1)]) for i in range(num_cells)]
        show_ellipses = precision is None

        return display_digits(num_cells, colors, text, top_text, show_ellipses, var_symbol)
    return (digits,)


@app.cell
def _(digits):
    chart1 = digits(1/3)
    chart1
    return


@app.cell
def _(digits):
    colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow', 'lightpink']

    chart2 = digits(0.431, 7, color=colors[1])
    chart2
    return (colors,)


@app.cell
def _(colors, decimal_to_binary, digits):
    precision = 5
    x_decimal = 0.321
    x_binary = decimal_to_binary(x_decimal, precision)

    chart3 = digits(x_binary, precision, color=colors[1])
    chart4 = digits(x_decimal, precision, color=colors[1])
    chart3
    return (chart4,)


@app.cell
def _(chart4):
    chart4
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
