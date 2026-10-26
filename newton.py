#!/usr/bin/env python3
"""Find a real zero of a mathematical expression with Newton's method."""

from __future__ import annotations

import argparse
import ast
import math
from collections.abc import Callable


SAFE_NAMES: dict[str, float | Callable[[float], float]] = {
    "abs": abs,
    "acos": math.acos,
    "asin": math.asin,
    "atan": math.atan,
    "cos": math.cos,
    "e": math.e,
    "exp": math.exp,
    "log": math.log,
    "pi": math.pi,
    "sin": math.sin,
    "sqrt": math.sqrt,
    "tan": math.tan,
}
ALLOWED_BINARY_OPERATORS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
ALLOWED_UNARY_OPERATORS = (ast.UAdd, ast.USub)


class ExpressionError(ValueError):
    """Raised when an expression contains syntax outside arithmetic."""


def compile_expression(expression: str) -> Callable[[float], float]:
    """Compile safe arithmetic expression containing variable ``x``."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise ExpressionError(f"invalid expression: {error.msg}") from error

    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and not isinstance(node.op, ALLOWED_BINARY_OPERATORS):
            raise ExpressionError("only +, -, *, / and ** operators are allowed")
        if isinstance(node, ast.UnaryOp) and not isinstance(node.op, ALLOWED_UNARY_OPERATORS):
            raise ExpressionError("only unary + and - operators are allowed")
        if isinstance(node, ast.Name) and node.id not in {*SAFE_NAMES, "x"}:
            raise ExpressionError(f"unknown name: {node.id}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in SAFE_NAMES:
                raise ExpressionError("only listed math functions may be called")
        if not isinstance(
            node,
            (
                ast.Expression,
                ast.BinOp,
                ast.UnaryOp,
                ast.Call,
                ast.Name,
                ast.Load,
                ast.Constant,
                *ALLOWED_BINARY_OPERATORS,
                *ALLOWED_UNARY_OPERATORS,
            ),
        ):
            raise ExpressionError(f"unsupported expression element: {type(node).__name__}")

    code = compile(tree, "<expression>", "eval")

    def function(x: float) -> float:
        try:
            # AST validation above limits eval to arithmetic and named math functions.
            return float(eval(code, {"__builtins__": {}}, {**SAFE_NAMES, "x": x}))
        except (ArithmeticError, TypeError, ValueError) as error:
            raise ExpressionError(f"cannot evaluate expression at x={x}: {error}") from error

    return function


def newton(
    function: Callable[[float], float],
    start: float,
    step: float,
    iterations: int,
    tolerance: float,
) -> tuple[float, int]:
    """Return ``(root, iterations_used)`` using a central-difference derivative."""
    x = start
    for iteration in range(1, iterations + 1):
        value = function(x)
        if abs(value) <= tolerance:
            return x, iteration - 1

        derivative = (function(x + step) - function(x - step)) / (2 * step)
        if not math.isfinite(derivative) or abs(derivative) <= math.ulp(1.0):
            raise ValueError(f"derivative is zero or non-finite near x={x}")

        next_x = x - value / derivative
        if not math.isfinite(next_x):
            raise ValueError("method produced a non-finite value")
        if abs(next_x - x) <= tolerance:
            return next_x, iteration
        x = next_x

    raise ValueError(f"did not converge after {iterations} iterations")


def positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--help", action="help", help="show this help message and exit")
    parser.add_argument("expression", help="function of x, e.g. 'x**2 - 2'")
    parser.add_argument("-x", "--start", type=float, default=1.0, help="start point (default: 1)")
    parser.add_argument(
        "-h",
        "--step",
        type=positive_float,
        default=1e-5,
        help="derivative step (default: 1e-5)",
    )
    parser.add_argument(
        "-n",
        "--iterations",
        type=positive_int,
        default=100,
        help="maximum iterations (default: 100)",
    )
    parser.add_argument(
        "-t",
        "--tolerance",
        type=positive_float,
        default=1e-10,
        help="convergence tolerance (default: 1e-10)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        function = compile_expression(args.expression)
        root, used = newton(
            function, args.start, args.step, args.iterations, args.tolerance
        )
    except (ExpressionError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error

    print(f"root = {root:.12g}")
    print(f"f(root) = {function(root):.3g}")
    print(f"iterations = {used}")


if __name__ == "__main__":
    main()
