"""
Static Single Assignment (SSA) transformations.
"""

from typing import Optional

from .define_use import DefineUse
from .verify import VerifyIR
from ..ir import *
from ..utils import Gensym

_Ctx = dict[str, str]

class _SSAInstance(DefaultTransformVisitor):
    """Single-use instance of an SSA pass."""
    gensym: Gensym

    def __init__(self, func: FunctionDef, names: set[str]):
        self.func = func
        self.gensym = Gensym(*names)

    def apply(self) -> FunctionDef:
        return self._visit(self.func, {})

    def _resolve(self, name: str, ctx: _Ctx) -> str:
        return ctx.get(name, name)

    def _visit_var(self, e: Var, ctx: _Ctx):
        return Var(self._resolve(e.name, ctx))

    def _visit_var_assign(self, stmt: VarAssign, ctx: _Ctx):
        # visit the expression
        e = self._visit(stmt.expr, ctx)

        # generate a new name if needed
        if stmt.var in ctx:
            name = self.gensym.fresh(stmt.var)
        else:
            name = stmt.var

        ctx = { **ctx, stmt.var: name }
        return VarAssign(name, stmt.ty, e), ctx

    def _visit_tuple_binding(self, vars: TupleBinding, ctx: _Ctx):
        new_vars: list[str | ir.TupleBinding] = []
        for name in vars:
            if isinstance(name, str):
                # generate a new name if needed
                if name in ctx:
                    t = self.gensym.fresh(name)
                    ctx = { **ctx, name: t }
                    new_vars.append(t)
                else:
                    ctx = { **ctx, name: name }
                    new_vars.append(name)
            elif isinstance(name, TupleBinding):
                elts, ctx = self._visit_tuple_binding(name, ctx)
                new_vars.append(elts)
            else:
                raise NotImplementedError('unexpected tuple identifier', name)
        return TupleBinding(new_vars), ctx

    def _visit_tuple_assign(self, e: TupleAssign, ctx: _Ctx):
        binding, ctx = self._visit_tuple_binding(e.binding, ctx)
        expr = self._visit(e.expr, ctx)
        return TupleAssign(binding, e.ty, expr), ctx

    def _visit_ref_assign(self, stmt, ctx):
        s = super()._visit_ref_assign(stmt, ctx)
        return s, ctx

    def _visit_return(self, stmt: Return, ctx: _Ctx):
        s = super()._visit_return(stmt, ctx)
        return s, ctx

    def _visit_block(self, block: Block, ctx: _Ctx):
        stmts: list[Stmt] = []
        for stmt in block.stmts:
            stmt, ctx = self._visit_statement(stmt, ctx)
            stmts.append(stmt)
        return Block(stmts), ctx

    def _visit_function(self, func: FunctionDef, ctx: _Ctx):
        ctx = ctx.copy()
        for arg in func.args:
            ctx[arg.name] = arg.name

        body, _ = self._visit_block(func.body, ctx)
        return FunctionDef(func.name, func.args, body, func.ty, func.ctx)

    # override to get typing hint
    def _visit(self, e, ctx: _Ctx):
        return super()._visit(e, ctx)

    # override to get typing hint
    def _visit_statement(self, stmt: Stmt, ctx: _Ctx) -> tuple[Stmt, _Ctx]:
        return super()._visit_statement(stmt, ctx)


class SSA:
    """
    Transformation pass to convert the IR to Static Single Assignment (SSA) form.

    This pass converts the IR to SSA form by introducing new variables for each
    assignment, ensuring that each variable is assigned exactly once.
    This transformation is generally used as a cleanup pass when previous
    transformations violate the SSA invariant of (valid) FPy IRs.
    """

    @staticmethod
    def apply(func: FunctionDef, names: Optional[set[str]] = None) -> FunctionDef:
        if names is None:
            uses = DefineUse.analyze(func)
            names = set(uses.keys())
        ir = _SSAInstance(func, names).apply()
        VerifyIR.check(ir)
        return ir
