"""FPy functions are the result of `@fpy` decorators."""

from abc import abstractmethod
from typing import Any, Optional
from titanfp.fpbench.fpcast import FPCore
from titanfp.arithmetic.evalctx import EvalCtx

from ..ir import FunctionDef
from ..frontend.fpc import fpcore_to_fpy

class Function:
    """
    FPy function.

    This object is created by the `@fpy` decorator and represents
    a function in the FPy runtime.
    """
    ir: FunctionDef
    env: dict[str, Any]
    rt: Optional['BaseInterpreter']

    def __init__(
        self,
        ir: FunctionDef,
        env: dict[str, Any],
        rt: Optional['BaseInterpreter'] = None
    ):
        self.ir = ir
        self.env = env
        self.rt = rt

    def __repr__(self):
        return f'{self.__class__.__name__}(ir={self.ir}, ...)'

    def __call__(self, *args, ctx: Optional[EvalCtx] = None):
        rt = get_default_interpreter() if self.rt is None else self.rt
        return rt.eval(self, args, ctx=ctx)

    @property
    def args(self):
        return self.ir.args

    @property
    def name(self):
        return self.ir.name

    @staticmethod
    def from_fpcore(core: FPCore):
        if not isinstance(core, FPCore):
            raise TypeError(f'expected FPCore, got {core}')
        ir = fpcore_to_fpy(core)
        return Function(ir, {})

    def with_rt(self, rt: 'BaseInterpreter'):
        if not isinstance(rt, BaseInterpreter):
            raise TypeError(f'expected BaseInterpreter, got {rt}')
        return Function(self.ir, self.env, rt)


class BaseInterpreter:
    """Abstract base class for FPy interpreters."""

    @abstractmethod
    def eval(self, func: Function, args, ctx: Optional[EvalCtx] = None):
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
