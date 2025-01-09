"""Module containing the interpreter for FPy programs."""

from typing import Any, Optional, Sequence

from titanfp.arithmetic.evalctx import EvalCtx
from titanfp.arithmetic.ieee754 import ieee_ctx
from titanfp.arithmetic.mpmf import MPMF, Interpreter as MPMFInterpreter
from titanfp.titanic.digital import Digital
from titanfp.titanic.ndarray import NDArray

from ..ir import *


ScalarVal = str | int | float | Digital
TensorVal = tuple | list | NDArray

class Interpreter(ReduceVisitor):
    """Interpreter for FPy programs."""
    _rt: MPMFInterpreter

    def __init__(self):
        self._rt = MPMFInterpreter

    # TODO: what are the semantics of arguments
    def _arg_to_mpmf(self, arg, ctx):
        if isinstance(arg, str | int | float):
            return MPMF(x=arg, ctx=ctx)
        elif isinstance(arg, Digital):
            return arg
        elif isinstance(arg, tuple | list):
            raise NotImplementedError()
        else:
            raise NotImplementedError(f'unknown argument type {arg}')

    def eval(self,
        func: Function,
        arg_seq: Sequence[Any],
        ctx: Optional[EvalCtx] = None
    ):
        if not isinstance(func, Function):
            raise TypeError(f'Expected Function, got {type(func)}')
        args = tuple(arg_seq)
        if len(args) != len(func.args):
            raise TypeError(f'Expected {len(func.args)} arguments, got {len(args)}')
        if ctx is None:
            ctx = ieee_ctx(11, 64)
        for val, arg in zip(args, func.args):
            match arg.ty:
                case AnyType():
                    ctx = ctx.let([(arg.name, self._arg_to_mpmf(val, ctx))])
                case RealType():
                    x = self._arg_to_mpmf(val, ctx)
                    if isinstance(x, Digital):
                        ctx = ctx.let([(arg.name, x)])
                    else:
                        raise NotImplementedError(f'argument is a scalar, got data {val}')
                case _:
                    raise NotImplementedError(f'unknown argument type {arg.ty}')
        return self._visit(func.body, ctx)

    def _visit_var(self, e, ctx: EvalCtx):
        if e.name not in ctx.bindings:
            raise RuntimeError(f'unbound variable {e.name}')
        return ctx.bindings[e.name]

    def _visit_decnum(self, e, ctx: EvalCtx):
        return MPMF(x=e.val, ctx=ctx)

    def _visit_integer(self, e, ctx: EvalCtx):
        return MPMF(x=e.val, ctx=ctx)

    def _visit_digits(self, e, ctx: EvalCtx):
        raise NotImplementedError

    def _visit_unknown(self, e, ctx: EvalCtx):
        raise NotImplementedError

    def _visit_nary_expr(self, e, ctx: EvalCtx):
        raise NotImplementedError

    def _visit_compare(self, e, ctx: EvalCtx):
        raise NotImplementedError

    def _visit_tuple_expr(self, e, ctx: EvalCtx):
        raise NotImplementedError

    def _visit_ref_expr(self, e, ctx: EvalCtx):
        raise NotImplementedError

    def _visit_comp_expr(self, e, ctx: EvalCtx):
        raise NotImplementedError

    def _visit_if_expr(self, e, ctx: EvalCtx):
        raise NotImplementedError

    def _visit_var_assign(self, stmt, ctx: EvalCtx):
        raise NotImplementedError

    def _visit_tuple_assign(self, stmt, ctx: EvalCtx):
        raise NotImplementedError

    def _visit_if1_stmt(self, stmt, ctx: EvalCtx):
        raise NotImplementedError

    def _visit_if_stmt(self, stmt, ctx: EvalCtx):
        raise NotImplementedError

    def _visit_while_stmt(self, stmt, ctx: EvalCtx):
        raise NotImplementedError

    def _visit_for_stmt(self, stmt, ctx: EvalCtx):
        raise NotImplementedError

    def _visit_return(self, stmt, ctx: EvalCtx):
        return self._visit(stmt.expr, ctx)

    def _visit_block(self, block, ctx: EvalCtx):
        for stmt in block.stmts:
            if isinstance(stmt, Return):
                return self._visit(stmt, ctx)
            else:
                ctx = self._visit(stmt, ctx)

    def _visit_function(self, func, ctx):
        raise NotImplementedError('do not call directly')


