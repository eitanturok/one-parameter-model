# Numerical Algorithms - 234125 - Second Exam - Winter Semester 2023

**Technion - Israel Institute of Technology**
**Faculty of Computer Science**
**Numerical Algorithms - 234125**
**Group A**

**Second Exam, Winter Semester 2023**
**March 16, 2023**

## Instructions:
- You may use a formula sheet of 4 double-sided pages (total 8 pages).
- Sharing materials between students during the exam is prohibited.
- You may use a simple calculator.
- Exam duration – 3 hours.
- This exam contains 15 multiple choice questions with varying weights totaling 101 points.
- The score for each question appears in square brackets at the beginning.
- You must answer all questions and mark the answers on the answer sheet.
- You must submit the exam booklet – write your ID number at the top of each page.

**Note: Keep the order of questions on the answer sheet.**

**Good luck!**

---

## Question 1 [7 points]
Given the matrices:
$$A = \begin{pmatrix} 0.8 & 0 \\ 0.6 & 1 \end{pmatrix}, \quad B = \begin{pmatrix} 0 & 0.8 \\ 1 & 0.6 \end{pmatrix}$$

Denote their QR decompositions as $A = Q_A R_A$, $B = Q_B R_B$, where $Q_A, Q_B$ are orthonormal matrices, and $R_A, R_B$ are upper triangular matrices with positive diagonal.

Does there exist a unique orthonormal matrix $G$ satisfying $GQ_A = Q_B$, and if so, what is it?

a. Cannot be determined since QR decompositions are not unique.
b. No such orthonormal $G$ exists.
c. Yes, $G = \begin{pmatrix} 0.6 & 0.8 \\ 0.8 & 0.6 \end{pmatrix}$
d. Yes, $G = \begin{pmatrix} 0.8 & 0.6 \\ 0.6 & 0.8 \end{pmatrix}$
e. Yes, $G = \begin{pmatrix} 0.6 & 0.8 \\ -0.8 & 0.6 \end{pmatrix}$

## Question 2 [6 points]
Given a vector $v$ and a collection of vectors $\{u_i\}$, all in $\mathbb{R}^n$. Let $v^*$ be the closest vector to $v$ (in Euclidean distance) that can be written as $v^* = \sum a_i u_i$. Mark the most correct statement:

a. The vector $v - v^*$ is orthogonal to $v^*$.
b. If vector $x$ is orthogonal to all vectors $u_j$, then $x$ is parallel to $v - v^*$.
c. None of the statements are correct.
d. For any pair of vectors $u_i, u_j \in \{u_i\}$, the vector $v - v^*$ is orthogonal to $u_j$.
e. The vector $v^*$ is orthogonal to all vectors $u_j$.

## Question 3 [7 points]
Given the matrix $\mathbf{A} = uv^T$ for some vectors $u \in \mathbb{R}^m, v \in \mathbb{R}^n$.

What can be said about the $L_1$ norm of the matrix?

a. $|||\mathbf{A}|||_1 = ||u||_1 ||v||_\infty$
b. $|||\mathbf{A}|||_1 = ||u||_1 ||v||_2$
c. $|||\mathbf{A}|||_1 = ||u||_1 ||v||_1$
d. $|||\mathbf{A}|||_1 = ||u||_\infty ||v||_\infty$
e. $|||\mathbf{A}|||_1 = ||u||_\infty ||v||_1$

## Question 4 [7 points]
For an iterative solution of the system of equations $Ax = b$ with real square non-singular matrix $A$, using the iterative formula $A^T A y = A^T b$, and solving the system $x = A^T y$ for some parameter, it is proposed to define the following:

$$y_{k+1} = y_k + \omega D^{-1}(A^T b - A^T A y_k)$$

where $D$ is a diagonal matrix whose elements are the diagonal elements of $A^T A$, and $\omega$ is a given scalar parameter.

We want to develop a formula of the form $x_{k+1} = Tx_k + c$, which will be equivalent to the above method, provided we choose appropriate $T, c$. Given that $x_0 = A^T y_0$, what are the appropriate $T, c$?

a. $T = I - \omega D^{-1} A^T A$, $c = \omega D^{-1} A^T b$
b. $T = I - \omega D^{-1} A^T A$, $c = A^T \omega D^{-1} b$
c. $T = I - \omega A^T D^{-1} A$, $c = \omega D^{-1} b$
d. $T = I - \omega A^T D^{-1} A$, $c = \omega D^{-1} A^T b$
e. $T = I - \omega A^T D^{-1} A$, $c = \omega A^T D^{-1} b$

## Question 5 [7 points]
Linear convolution of row vector $v$ with itself gives:
$$v \otimes v = (1, 8, 12, 22, 20, 12, 9)$$

What is $v$?

a. $v = (1, 4, 2, 3)$
b. Two of the presented solutions are possible
c. $v = (-1, 4, 2, 3)$
d. $v = (1, -4, 2, 3)$
e. All presented solutions are incorrect

## Question 6 [7 points]
Given the matrix $A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 8 & 12 \\ 2 & 4 & 6 \end{bmatrix}$, find its singular values.

a. $\sqrt{294}, 0, 0$
b. $12, 24, 6$
c. $\sqrt{294}, 146, 42$
d. $\sqrt{42}, 15, 84$
e. $42, 0, 0$
f. $\sqrt{146}, 0, 0$

## Question 7 [7 points]
Which of the following statements is NOT correct?

a. Denote the QR decomposition of $A$ as $A = QR$, then the Cholesky decomposition of matrix $A^T A$ is given by $R^T R$.

b. If $A^2 = I$ for real matrix $A$, then it is certainly a symmetric and orthonormal matrix.

c. Denote by $|||\cdot|||_p$ the induced matrix norm $L_p$ of real symmetric matrix $A$, and for all $p \geq 1$, then for all $A$ we have $|||A|||_2 \leq |||A|||_p$.

d. The spectral radius of a real matrix $A$ equals 0 if and only if all elements of the matrix are zero.

e. If $A$ is a real, upper triangular, and non-singular matrix, then the Gauss-Seidel method for solving the system of equations $Ax = b$ converges with certainty.

## Question 8 [7 points]
Given a matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ such that $\text{rank}(\mathbf{A}) \geq 2$. Let $p = \text{rank}(\mathbf{A}) - 2$ and let
$$\mathbf{B}^* = \arg\min_{\mathbf{B} \in \mathbb{R}^{m \times n}} \{||\mathbf{A} - \mathbf{B}||_F^2 | \text{rank}(\mathbf{B}) = p\}$$
be the rank-$p$ matrix closest to $\mathbf{A}$ according to Frobenius norm.

Which statement is necessarily true about the difference matrix $\mathbf{A} - \mathbf{B}^*$?

a. $|||\mathbf{A} - \mathbf{B}^*|||_2^2 = |||\mathbf{A} - \mathbf{B}^*|||_F^2$
b. $\exists u \in \mathbb{R}^m, v \in \mathbb{R}^n: \mathbf{A} - \mathbf{B}^* = uv^T$
c. $||\frac{1}{n}(\mathbf{A} - \mathbf{B}^*)1||_2^2 = 2$
d. $\exists \mathbf{G} \in \mathbb{R}^{m \times 2}, \mathbf{C} \in \mathbb{R}^{2 \times n}: \mathbf{A} - \mathbf{B}^* = \mathbf{G}\mathbf{C}$
e. $\mathbf{A}^T(\mathbf{A} - \mathbf{B}^*) = \mathbf{0}$

## Question 9 [6 points]
Given a regularized least squares problem with matrices $\mathbf{A} \in \mathbb{R}^{m \times n}$, $\mathbf{C} \in \mathbb{R}^{n \times n}$ and regularization coefficient $\lambda > 0$:
$$\min_{x \in \mathbb{R}^n} \{||\mathbf{A}x - b||_2^2 + \lambda ||\mathbf{C}x - d||_2^2\}$$

Which of the following statements is necessarily true?

a. If $\mathbf{A}^T\mathbf{A} + \lambda\mathbf{C}^T\mathbf{C}$ is not positive definite, then there are necessarily infinitely many solutions to the problem.

b. If the smallest singular value of $\mathbf{A}$ is greater than or equal to $\lambda$, then there are infinitely many solutions to the problem.

c. If $b$ belongs to the column space of $\mathbf{A}$, then there are necessarily infinitely many solutions to the problem.

d. If $n < m$, then there are necessarily infinitely many solutions to the problem.

e. If $\mathbf{A}$ is square and regular, then there are necessarily infinitely many solutions to the problem.

## Question 10 [7 points]
For a matrix $\mathbf{T} \in \mathbb{R}^{m \times n}$, it holds that:
$$|||\mathbf{T}|||_2 \leq \underbrace{|||\mathbf{T}|||_F}_{\text{from lecture}} \leq c \cdot |||\mathbf{T}|||_2$$

Find the tight bound $c$.

a. $n$
b. $\sqrt{n}$
c. $\sqrt{\min\{m,n\}}$
d. $\min\{m,n\}$
e. $m$

## Question 11 [7 points]
Given a circulant and symmetric matrix $\mathbf{A}$ with first row $c^T = \frac{1}{4}[-3, 3, 1, 3]$.

Define matrix $\mathbf{B} = \mathbf{A}^{-1} + 2\mathbf{I}$. Calculate $|||\mathbf{B}|||_2$.

Hint: The 4×4 Fourier matrix is given by $\mathbf{F} = \frac{1}{2}\begin{bmatrix} 1 & 1 & 1 & 1 \\ 1 & -i & -1 & i \\ 1 & -1 & 1 & -1 \\ 1 & i & -1 & -i \end{bmatrix}$

a. $|||\mathbf{B}|||_2 = 2.25$
b. $|||\mathbf{B}|||_2 = 3.5$
c. $|||\mathbf{B}|||_2 = 10$
d. $|||\mathbf{B}|||_2 = 3$
e. $|||\mathbf{B}|||_2 = 1$

## Question 12 [7 points]
Given the following matrices:
$$\mathbf{A} = \begin{bmatrix} 1 & 1 & 2 \\ 1 & 0 & -2 \\ -1 & 2 & 3 \end{bmatrix}, \quad \mathbf{B} = \begin{bmatrix} 1 & 1 & 2 & 2 \\ 1 & 0 & 1 & -2 \\ -1 & 2 & 1 & 3 \end{bmatrix}$$

$$\mathbf{Q}_\mathbf{A} = \begin{bmatrix} 1 & 4 & 2 \\ 1 & 1 & -3 \\ -1 & 5 & -1 \end{bmatrix} \begin{bmatrix} 1/\sqrt{3} & 0 & 0 \\ 0 & 1/\sqrt{42} & 0 \\ 0 & 0 & 1/\sqrt{14} \end{bmatrix}$$

Given that $\mathbf{Q}_\mathbf{A}$ was obtained from the Gram-Schmidt process on $\mathbf{A}$. Which matrix might be obtained from the Gram-Schmidt process on $\mathbf{B}$?

[Multiple choice options with various matrices - detailed matrix options provided in original]

## Question 13 [6 points]
Given a regularized LS problem $\min_x \{||\mathbf{A}x - b||_2^2 + \lambda ||\mathbf{C}x - d||_2^2\}$ with coefficient $\lambda > 0$.

Mark the statement that is necessarily true.

a. The solution to the problem is necessarily unique, for any $\mathbf{A}, \mathbf{C}$ chosen.
b. If the solution to the problem is unique, then necessarily $\mathbf{A}$ has linearly independent columns.
c. If the solution to the problem is unique, then necessarily vector $b$ belongs to the column space of $\mathbf{A}$.
d. If the solution to the problem is unique, then necessarily $\mathbf{A}^T\mathbf{A}$ is positive definite.
e. If the solution to the problem is unique, then necessarily the block matrix $\begin{bmatrix} \mathbf{A} \\ \sqrt{2\lambda}\mathbf{C} \end{bmatrix}$ has linearly independent columns.

## Question 14 [7 points]
Given two orthonormal matrices $\mathbf{U}, \mathbf{V}$ of size $10 \times 10$.

Denote their columns as $v_1, v_2, \ldots, v_{10}$ and $u_1, u_2, \ldots, u_{10}$ respectively. Define matrices $\mathbf{A}$ and $\mathbf{B}$ as follows:
$$\mathbf{A} = \sum_{i=2,4,8,10} u_i v_i^T, \quad \mathbf{B} = \sum_{i=1,2,3,4} u_i v_i^T$$

Mark the correct statement.

a. $||\mathbf{A} + \mathbf{B}||_2^2 = 12$
b. $||\mathbf{A} + \mathbf{B}||_2^2 = 11$
c. $||\mathbf{A} + \mathbf{B}||_2^2 = 9$
d. $||\mathbf{A} + \mathbf{B}||_2^2 = 8$
e. $||\mathbf{A} + \mathbf{B}||_2^2 = 10$

## Question 15 [6 points]
Given the matrix $\begin{pmatrix} 4 & 1 & 2 \\ 2 & x & y \\ 1 & 4 & 7 \end{pmatrix}$. The matrix undergoes LU decomposition with partial pivoting.

Under what condition will there be row permutation? (iff means if and only if)

Choose the most correct answer.

a. Need permutation iff $x = 0$ or $y = 0$.
b. Need permutation iff $x = 0$.
c. Need permutation iff $|x| > 4$.
d. Need permutation under another condition.
e. Need permutation iff $|x| > 4$ or $|y| > 7$.
