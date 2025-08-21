# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
def _first_derivative(f,x,h):
    """
    Estimate the first derivative f'(x) with the following parameters:
    f: the function to differentiate
    x: the point at which to estimate the derivative
    h: a small step size
    """
    return (f(x + h) - f(x - h)) / (2.0 * h)

def _second_derivative(f, x, h):
    """
    Estimate the second derivative f''(x)
    """
    return (f(x + h) - 2.0 * f(x) + f(x - h)) / (h**2)

def optimize(x0, f, tol=1e-4, max_iter=100, h=1e-3, return_history=False, comments=False):
    """
    Minimize a univariate function f(x) using Newton's method with the following parameters:
    
    =Parameters=
    x0 : Starting point for the iterations.
    f : The function to minimize. .
    tol : Tolerance for stopping. We stop when the change |x_{k+1} - x_k| is below tol.
    max_iter : Maximum number of iterations to attempt.
    h : Step size for finite-difference derivatives. 
    return_history : If True, also return a list of iterates [x0, x1, ..., x*] so you can inspect the path.
    verbose : bool, optional
        If True, print progress each iteration (x, f(x), f'(x), f''(x)).

    =Return=
    x_star : The estimated minimizer (location of minimum).
    hist : Only returned if return_history=True. The sequence of x values visited.
    """

    x = float(x0)     
    history = [x]

    for it in range(1, max_iter + 1):
        # 1) Compute first and second derivative at current point x
        fprime = _first_derivative(f, x, h)
        fsecond = _second_derivative(f, x, h)

        if comments:
            print(f"[iter {it:02d}] x={x:.8f}  f(x)={f(x):.8f}  f'(x)={fprime:.8e}  f''(x)={fsecond:.8e}")

        if abs(fsecond) < 1e-10:
            if commentss:
                print("Second derivative too small; stopping early.")
            break

        x_new = x - fprime / fsecond

        if abs(x_new - x) < tol:
            x = x_new
            history.append(x)
            if comments:
                print(f"Converged: |Δx| < tol ({tol})")
            break
            
        x = x_new
        history.append(x)
        
    return (x, history) if return_history else x

# %%
