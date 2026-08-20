# Robust Index Tracking Using HMCR

## Model

### Robust Portfolio Optimization with Wasserstein Distance

We consider the following distributionally robust portfolio optimization problem:

$$
\inf_{\omega \in W}
\sup_{P \in \mathcal{P}_{\epsilon}(\hat P)}
\operatorname{HMCR}_{\alpha}(f(\omega,\xi)),
$$
where
$$
W = \{ \omega \in \mathbb R^m : \omega^\top \mathbf 1 = 1, \ \omega \ge 0 \},
$$
and
$$
\mathcal P_\epsilon (\hat P) = \{ P: d_W(P, \hat P) \le \epsilon \}.
$$

Here,
* $d_W$ denotes the 1-Wasserstein distance,
* $\hat P$ denotes the empirical distribution,
* $\epsilon$ is the radius of the Wasserstein ambiguity set.

The portfolio loss function is defined as
$$
f(\omega,\xi) = -\langle \omega,\xi\rangle.
$$

### LASSO Constraint

For the LASSO-constrained model, we replace $W$ with
$$
W_L = \{ \omega \in \mathbb R^m : \omega^\top \mathbf 1 = 1, \ \|\omega\|_1 \le s \}
$$
where $s$ controls the $\ell_1$-norm of the portfolio weights.

### Robust Return Constraint

For the return-constrained model, we additionally impose
$$
\sup_{Q \in \mathcal P_\epsilon (\hat P)} \mathbb E_Q [f(\omega, \xi)] \le \mathbb E_{\hat P} [f(\omega, \xi)]
$$

This constraint controls the portfolio loss under distributions within the Wasserstein ambiguity set relative to the empirical distribution.

## How to Use

### 1. Python Version

This project was tested with

```bash
python --version

# Python 3.11.9
```

### 2. Install Dependencies

Create and activate a virtual environment, then install the required packages:

```bash
pip install -r requirements.txt
```

### 3. Run the Experiments

Open `run.ipynb` and execute the notebook blocks in order.

If the S&P 500 data have not been downloaded yet, first run the `download` block. The downloaded data will be saved in the `data` directory located one level above the directory containing the code.

Then, run the `calculate` block to solve the optimization problems and generate the portfolio results. The results will also be saved in the `data` directory located one level above the directory containing the code.

Finally, run the `plot` block to visualize the backtest results and portfolio weights.