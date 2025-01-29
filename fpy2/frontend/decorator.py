"""
Decorators for the FPy language.
"""

import inspect

from typing import Callable, Optional
from typing import ParamSpec, TypeVar, overload

from .codegen import IRCodegen
from .definition import DefinitionAnalysis
from .fpyast import FunctionDef
from .live_vars import LiveVarAnalysis
from .parser import Parser
from .syntax_check import SyntaxCheck

from ..passes import SSA, VerifyIR
from ..runtime import Function

P = ParamSpec('P')
R = TypeVar('R')

@overload
def fpy(func: Callable[P, R]) -> Callable[P, R]:
    ...

@overload
def fpy(**kwargs) -> Callable[[Callable[P, R]], Callable[P, R]]:
    ...

def fpy(
    func: Optional[Callable[P, R]] = None, **kwargs
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator to parse a Python function into FPy.

    Constructs an FPy `Function` from a Python function.
    FPy is a stricter subset of Python, so this decorator will reject
    any function that is not valid in FPy.
    """
    if func is None:
        return lambda func: _apply_decorator(func, **kwargs)
    else:
        return _apply_decorator(func, **kwargs)


def _apply_decorator(func: Callable[P, R], **kwargs):
    # read the original source of the function
    sourcename = inspect.getabsfile(func)
    lines, start_line = inspect.getsourcelines(func)
    source = ''.join(lines)

    # parse the source as an FPy function
    ast = Parser(sourcename, source, start_line).parse()
    assert isinstance(ast, FunctionDef), "must be a function"

    # add context information
    ast.ctx = { **kwargs }

    # print(ast.format())

    # analyze and lower to the IR
    SyntaxCheck.analyze(ast)
    DefinitionAnalysis.analyze(ast)
    LiveVarAnalysis.analyze(ast)
    ir = IRCodegen.lower(ast)
    ir = SSA.apply(ir)
    VerifyIR.check(ir)

    # wrap the IR in a Function
    return Function(ir, func.__globals__)
