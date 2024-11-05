from typing import Callable

import torch

def atanh(y: torch.Tensor) -> torch.Tensor:
    """
    Compute the inverse hyperbolic tangent.
    """
    eps = torch.finfo(y.dtype).eps
    clamped_y = torch.clamp(y, -1.0 + eps, 1.0 - eps)
    return 0.5 * (clamped_y.log1p() - (-clamped_y).log1p())

def bmv(mat, vec):
    """
    Multiply a matrix by a vector (batched).
    """
    return torch.bmm(mat, vec.unsqueeze(-1)).squeeze(-1)

# FIXME: check poly and inv_poly
def poly(coeffs: torch.Tensor, x: torch.Tensor, increasing_order: bool = False) -> torch.Tensor:
    """
    Evaluate the polynomial at x.
    """
    batch_size, data_dim = x.shape[0], x.shape[1]

    if coeffs.ndim == 1:
        coeffs = coeffs.unsqueeze(0).expand(batch_size, -1)
    else:
        assert coeffs.shape[0] == batch_size, "Coefficients must have same batch size as x."

    # Generate the powers of x using broadcasting
    powers = torch.arange(coeffs.shape[-1], dtype=x.dtype, device=x.device) # (1, degree + 1)
    if not increasing_order:
        powers = torch.flip(powers, dims=[-1])

    x_powers = x.unsqueeze(-1).pow(powers) # (batch_size, data_dim, degree + 1)
    # Calculate the polynomial values using broadcasting and sum along the last dimension
    y = (coeffs.unsqueeze(1) * x_powers).sum(dim=-1)

    return y

def inv_poly(coeffs: torch.Tensor, y: torch.Tensor, increasing_order: bool = False) -> torch.Tensor:
    """
    Compute the inverse of a polynomial given coefficients and a target y value.
    Returns the x value(s).
    """
    batch_size, data_dim = y.shape[0], y.shape[1]

    if coeffs.ndim == 1:
        coeffs = coeffs.unsqueeze(0).expand(batch_size, -1)
    else:
        assert coeffs.shape[0] == batch_size, "Coefficients must have same batch size as x."

    degree = coeffs.shape[-1] - 1

    if degree == 1:
        # ax+b=y => x=(y-b)/a
        if increasing_order:
            a, b = coeffs.split(1, dim=-1)
        else:
            b, a = coeffs.split(1, dim=-1)
        return (y - b) / a
    elif degree == 2:
        # ax^2+bx+c=y => x=(-b+sqrt(b^2-4ac))/2a
        if increasing_order:
            c, b, a = coeffs.split(1, dim=-1)
        else:
            a, b, c = coeffs.split(1, dim=-1)
        return (-b + torch.sqrt(b**2 - 4*a*(c - y))) / (2*a)
    else:
        raise NotImplementedError("Only implemented for degree 1 and 2.")
    
def polyder(coeffs: torch.Tensor, order: int = 1, increasing_order: bool = False) -> torch.Tensor:
    """
    Compute the derivative of a polynomial of a given order.

    Parameters:
    - coeffs: torch.Tensor - Coefficients of the polynomial
    - order: int - Order of the derivative

    Returns:
    - torch.Tensor - Coefficients of the derived polynomial
    """
    # Validate inputs
    if order < 0:
        raise ValueError("Order of derivative must be non-negative")
    
    derived_coeffs = coeffs.clone()
    for _ in range(order):
        # Multiply each coefficient by its power
        if not increasing_order:
            powers = torch.arange(derived_coeffs.shape[-1] - 1, -1, -1, 
                                dtype=derived_coeffs.dtype, device=derived_coeffs.device)
            derived_coeffs = derived_coeffs * powers
            derived_coeffs = derived_coeffs[..., :-1]
        else:
            powers = torch.arange(derived_coeffs.shape[-1], 
                                  dtype=derived_coeffs.dtype, device=derived_coeffs.device)
            derived_coeffs = derived_coeffs * powers
            derived_coeffs = derived_coeffs[..., 1:]

    return derived_coeffs

def skew_matrix(vec):
    """
    Convert a vector to a skew-symmetric matrix.
    """
    assert vec.shape[-1] == 3, "Input must has shape (..., 3)."
    zeros = torch.zeros_like(vec)
    return torch.cat([
        zeros[..., 0], -vec[..., 2], vec[..., 1],
        vec[..., 2], zeros[..., 1], -vec[..., 0],
        -vec[..., 1], vec[..., 0], zeros[..., 2]
    ], dim=-1).reshape(-1, 3, 3)

if __name__ == "__main__":
    c = torch.randn(1, 3)
    x = torch.randn(1, 4)
    # y = torch.randn(16, 4)

    y = poly(c, x)
    xx = inv_poly(c, y)
    yy = poly(c, xx)

    error_x = torch.abs(x - xx)
    error_y = torch.abs(y - yy)

    print('c', c)
    print('x', x)
    print('y', y)
    print('xx', xx)
    print('yy', yy)

    print('ex', error_x)
    print('ey', error_y)

    dc = polyder(c, 1, increasing_order=True)
    ddc = polyder(c, 2)

    print('dc', dc)
    print('ddc', ddc)