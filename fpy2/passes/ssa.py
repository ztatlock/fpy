"""
Static Single Assignment (SSA) transformation pass.
"""

from .reaching_defs import ReachingDefs, Reach
from .verify import VerifyIR
from ..ir import *
from ..utils import Gensym

_Ctx = dict[str, str]

class _SSAInstance(DefaultTransformVisitor):
    # class _SSAInstance(DefaultTransformVisitor):
    """Single-use instance of an SSA pass."""
    func: FunctionDef
    gensym: Gensym
    reaches: dict[Block, Reach]

    def __init__(
        self,
        func: FunctionDef,
        reaches: dict[Block, Reach]
    ):
        self.func = func
        self.gensym = Gensym()
        self.reaches = reaches

    def apply(self) -> FunctionDef:
        return self._visit(self.func, {})

    # override to get typing hint
    def _visit(self, e, ctx):
        return super()._visit(e, ctx)

    def _visit_var(self, e, ctx: _Ctx):
        return Var(ctx[e.name])

    def _visit_comp_expr(self, e, ctx: _Ctx):
        iterables = [self._visit(iterable, ctx) for iterable in e.iterables]

        ctx = ctx.copy()
        vars: list[str] = []
        for var in e.vars:
            name = self.gensym.fresh(var)
            vars.append(name)
            ctx[var] = name

        elt = self._visit(e.elt, ctx)
        return CompExpr(vars, iterables, elt)

    def _visit_var_assign(self, stmt: VarAssign, ctx: _Ctx):
        # visit the expression
        e = self._visit(stmt.expr, ctx)

        # generate a new name if needed
        t = self.gensym.fresh(stmt.var)
        ctx = { **ctx, stmt.var: t }
        return VarAssign(t, stmt.ty, e), ctx

    def _visit_tuple_binding(self, vars: TupleBinding, ctx: _Ctx):
        new_vars: list[str | TupleBinding] = []
        for name in vars:
            if isinstance(name, str):
                # generate a new name if needed
                t = self.gensym.fresh(name)
                ctx = { **ctx, name: t }
                new_vars.append(t)
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

    def _visit_ref_assign(self, stmt, ctx: _Ctx):
        var = ctx[stmt.var]
        slices = [self._visit(slice, ctx) for slice in stmt.slices]
        expr = self._visit(stmt.expr, ctx)
        return RefAssign(var, slices, expr), ctx

    def _visit_if1_stmt(self, stmt, ctx: _Ctx):
        # visit condition
        cond = self._visit(stmt.cond, ctx)
        body, body_ctx = self._visit_block(stmt.body, ctx)

        # update existing phi nodes
        new_phis: list[PhiNode] = []
        new_ctx = ctx.copy()
        for phi in stmt.phis:
            del new_ctx[phi.lhs]
            if phi.lhs in ctx:
                # TODO: infer type
                t = self.gensym.fresh(phi.name)
                lhs = ctx[phi.lhs]
                rhs = body_ctx[phi.rhs]
                phi = PhiNode(t, lhs, rhs, AnyType())
                new_phis.append(phi)
                new_ctx[phi.name] = t

        # create new phi variables
        for var in ctx:
            lhs = ctx[var]
            rhs = body_ctx[var]
            if lhs != rhs:
                # TODO: infer type
                t = self.gensym.fresh(var)
                phi = PhiNode(t, lhs, rhs, AnyType())
                new_phis.append(phi)
                new_ctx[var] = t

        s = If1Stmt(cond, body, new_phis)
        return s, new_ctx

    def _visit_if_stmt(self, stmt, ctx: _Ctx):
        # visit condition and branches
        cond = self._visit(stmt.cond, ctx)
        ift, ift_ctx = self._visit_block(stmt.ift, ctx)
        iff, iff_ctx = self._visit_block(stmt.iff, ctx)
        merged_vars = ift_ctx.keys() & iff_ctx.keys()

        # update existing phi nodes
        new_phis: list[PhiNode] = []
        new_ctx = ctx.copy()
        for phi in stmt.phis:
            if phi.lhs in ift_ctx and phi.rhs in iff_ctx:
                # TODO: infer type
                t = self.gensym.fresh(phi.name)
                lhs = ift_ctx[phi.lhs]
                rhs = iff_ctx[phi.rhs]
                phi = PhiNode(t, lhs, rhs, AnyType())
                new_phis.append(phi)
                new_ctx[phi.name] = t

        # create new phi variables
        for var in merged_vars:
            lhs = ift_ctx[var]
            rhs = iff_ctx[var]
            if lhs != rhs:
                # TODO: infer type
                t = self.gensym.fresh(var)
                phi = PhiNode(t, lhs, rhs, AnyType())
                new_phis.append(phi)
                new_ctx[var] = t

        s = IfStmt(cond, ift, iff, new_phis)
        return s, new_ctx

    def _visit_while_stmt(self, stmt, ctx: _Ctx):
        # compute variables requiring phi node
        reach = self.reaches[stmt.body]
        updated = ctx.keys() & reach.kill_out

        # create loop context with existing phi names
        loop_ctx = ctx.copy()
        for phi in stmt.phis:
            t = self.gensym.fresh(phi.name)
            loop_ctx[phi.name] = t
            del loop_ctx[phi.lhs]

        # add new phi names to loop context
        for var in updated:
            t = self.gensym.fresh(var)
            loop_ctx[var] = t

        # visit condition and body
        cond = self._visit(stmt.cond, loop_ctx)
        body, body_ctx = self._visit_block(stmt.body, loop_ctx)

        # update existing phi nodes
        new_phis: list[PhiNode] = []
        new_ctx = ctx.copy()
        for phi in stmt.phis:
            del new_ctx[phi.lhs]
            if phi.lhs in ctx:
                # TODO: infer type
                t = loop_ctx[phi.name]
                lhs = ctx[phi.lhs]
                rhs = body_ctx[phi.rhs]
                phi = PhiNode(t, lhs, rhs, AnyType())
                new_phis.append(phi)
                new_ctx[phi.name] = t

        # create new phi variables
        for var in updated:
            lhs = ctx[var]
            rhs = body_ctx[var]
            if lhs != rhs:
                # TODO: infer type
                t = loop_ctx[var]
                lhs = ctx[var]
                rhs = body_ctx[var]
                phi = PhiNode(t, lhs, rhs, AnyType())
                new_phis.append(phi)
                new_ctx[var] = t

        s = WhileStmt(cond, body, new_phis)
        return s, new_ctx

    def _visit_for_stmt(self, stmt, ctx: _Ctx):
        # visit iterable
        iterable = self._visit(stmt.iterable, ctx)
        iter_name = self.gensym.fresh(stmt.var)
        ctx = { **ctx, stmt.var: iter_name }

        # compute variables requiring phi node
        reach = self.reaches[stmt.body]
        updated = ctx.keys() & reach.kill_out

        # create loop context with existing phi names
        loop_ctx = ctx.copy()
        for phi in stmt.phis:
            t = self.gensym.fresh(phi.name)
            loop_ctx[phi.name] = t
            del loop_ctx[phi.lhs]

        # add new phi names to loop context
        for var in updated:
            t = self.gensym.fresh(var)
            loop_ctx[var] = t

        # visit body
        body, body_ctx = self._visit_block(stmt.body, loop_ctx)

        # update existing phi nodes
        new_phis: list[PhiNode] = []
        new_ctx = ctx.copy()
        for phi in stmt.phis:
            del new_ctx[phi.lhs]
            if phi.lhs in ctx:
                # TODO: infer type
                t = loop_ctx[phi.name]
                lhs = ctx[phi.lhs]
                rhs = body_ctx[phi.rhs]
                phi = PhiNode(t, lhs, rhs, AnyType())
                new_phis.append(phi)
                new_ctx[phi.name] = t

        # create new phi variables
        for var in updated:
            lhs = ctx[var]
            rhs = body_ctx[var]
            if lhs != rhs:
                # TODO: infer type
                t = loop_ctx[var]
                lhs = ctx[var]
                rhs = body_ctx[var]
                phi = PhiNode(t, lhs, rhs, AnyType())
                new_phis.append(phi)
                new_ctx[var] = t

        s = ForStmt(iter_name, stmt.ty, iterable, body, new_phis)
        return s, new_ctx

    def _visit_context(self, stmt, ctx: _Ctx):
        # TODO: what to do about `stmt.name`
        body, body_ctx = self._visit_block(stmt.body, ctx)
        return ContextStmt(stmt.name, stmt.props, body), body_ctx

    def _visit_return(self, stmt, ctx):
        s = Return(self._visit(stmt.expr, ctx))
        return s, ctx

    def _visit_phis(self, phis: list[PhiNode], lctx: _Ctx, rctx: _Ctx):
        raise NotImplementedError

    def _visit_loop_phis(self, phis: list[PhiNode], lctx: _Ctx, rctx: Optional[_Ctx]):
        raise NotImplementedError

    def _visit_function(self, func, ctx: _Ctx):
        ctx = ctx.copy()
        for arg in func.args:
            self.gensym.reserve(arg.name)
            ctx[arg.name] = arg.name

        body, _ = self._visit(func.body, ctx)
        return FunctionDef(func.name, func.args, body, func.ty, func.ctx)

    # override to get typing hint
    def _visit_block(self, block, ctx: _Ctx) -> tuple[Block, _Ctx]:
        return super()._visit_block(block, ctx)

class SSA:
    """
    Transformation pass to convert the IR to Static Single Assignment (SSA) form.

    This pass converts the IR to SSA form by introducing new variables for each
    assignment, ensuring that each variable is assigned exactly once.
    This transformation is generally used as a cleanup pass when previous
    transformations violate the SSA invariant of (valid) FPy IRs.
    """

    @staticmethod
    def apply(func: FunctionDef) -> FunctionDef:
        reaches = ReachingDefs.analyze(func)
        func = _SSAInstance(func, reaches).apply()
        VerifyIR.check(func)
        return func
