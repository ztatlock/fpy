"""FPy functions are the result of `@fpy` decorators."""

from abc import abstractmethod
from typing import Any, Optional
from titanfp.arithmetic.evalctx import EvalCtx

from ..ir import FunctionDef

class Function:
    """
    FPy function.

    This object is created by the `@fpy` decorator and represents
    a function in the FPy runtime.
    """
    func: FunctionDef
    env: dict[str, Any]
    rt: Optional['BaseInterpreter']

    def __init__(
        self,
        func: FunctionDef,
        env: dict[str, Any],
        rt: Optional['BaseInterpreter'] = None
    ):
        self.func = func
        self.env = env
        self.rt = rt

    def __call__(self, *args, ctx: Optional[EvalCtx] = None):
        rt = get_default_interpreter() if self.rt is None else self.rt
        return rt.eval(self.func, args, ctx=ctx)

    def with_rt(self, rt: 'BaseInterpreter'):
        if not isinstance(rt, BaseInterpreter):
            raise TypeError(f'expected BaseInterpreter, got {rt}')
        return Function(self.func, self.env, rt)


class BaseInterpreter:
    """
    Abstract base class for FPy interpreters.

    Evaluates `Function` objects.
    """

    @abstractmethod
    def eval(self, func: FunctionDef, args, ctx: Optional[EvalCtx] = None):
        raise NotImplementedError('virtual method')

_default_interpreter: Optional[BaseInterpreter] = None

def get_default_interpreter() -> BaseInterpreter:
    """Get the default FPy interpreter."""
    global _default_interpreter
    if _default_interpreter is None:
        raise RuntimeError('no default interpreter available')
    return _default_interpreter

def set_default_interpreter(rt: BaseInterpreter):
    """Sets the default FPy interpreter"""
    global _default_interpreter
    if not isinstance(rt, BaseInterpreter):
        raise TypeError(f'expected BaseInterpreter, got {rt}')
    _default_interpreter = rt
