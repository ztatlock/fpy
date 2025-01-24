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
        new_vars: list[str | TupleBinding] = []
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
        expr = self._visit(e.expr, ctx)
        binding, ctx = self._visit_tuple_binding(e.binding, ctx)
        return TupleAssign(binding, e.ty, expr), ctx

    def _visit_phis(self, phis, lctx: _Ctx, rctx: _Ctx):
        # visting a merge point after visiting both branches
        # generate a new name if needed
        phi_names: dict[str, str] = {}
        for phi in phis:
            if phi.name in lctx or phi.name in rctx:
                name = self.gensym.fresh('t')
                phi_names[phi.name] = name
            else:
                phi_names[phi.name] = phi.name

        # create new phi variables
        new_phis: list[PhiNode] = []
        for phi in phis:
            name = phi_names[phi.name]
            lhs = self._resolve(phi.lhs, lctx)
            rhs = self._resolve(phi.rhs, rctx)
            new_phis.append(PhiNode(name, lhs, rhs, phi.ty))

        # merge contexts
        ctx = { **lctx, **rctx }
        for name in phi_names:
            ctx[name] = phi_names[name]

        print(phis, lctx, rctx, new_phis, ctx)
        return new_phis, ctx

    def _visit_loop_phis(self, phis, lctx: _Ctx, rctx: Optional[_Ctx]):
        new_phis: list[PhiNode] = []
        if rctx is None:
            # visiting join point before visiting loop body
            phi_names: dict[str, str] = {}
            for phi in phis:
                if phi.name in lctx:
                    name = self.gensym.fresh('t')
                    phi_names[phi.name] = name
                else:
                    phi_names[phi.name] = phi.name

            # create new phi variables
            for phi in phis:
                name = phi_names[phi.name]
                lhs = self._resolve(phi.lhs, lctx)
                new_phis.append(PhiNode(name, lhs, phi.rhs, phi.ty))

            # create context
            ctx = { **lctx }
            for name in phi_names:
                ctx[name] = phi_names[name]
        else:
            # re-visiting join point after visiting loop body
            # only `rctx` will be different

            # create new phi variables
            for phi in phis:
                name = phi.name
                lhs = self._resolve(phi.lhs, lctx)
                rhs = self._resolve(phi.rhs, rctx)
                new_phis.append(PhiNode(name, lhs, rhs, phi.ty))

            # merge contexts
            ctx = { **lctx, **rctx }

        print(phis, lctx, rctx, new_phis, ctx)
        return new_phis, ctx

    # override to get typing hint
    def _visit(self, e, ctx: _Ctx):
        return super()._visit(e, ctx)


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
        print(func)
        func = _SSAInstance(func, names).apply()
        print(func)
        VerifyIR.check(func)
        return func
