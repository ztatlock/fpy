"""Live variable analysis for the FPy AST."""

from .fpyast import *
from .visitor import AstVisitor

_LiveSet = set[str]

class LiveVarAnalysisInstance(AstVisitor):
    """Single-use live variable analyzer"""

    def analyze(self, func: FunctionDef):
        """Analyze the live variables in a function."""
        if not isinstance(func, FunctionDef):
            raise TypeError(f'expected a Function, got {func}')
        self._visit_function(func, set())

    def _visit_var(self, e, ctx) -> _LiveSet:
        live = { e.name }
        e.attribs[LiveVarAnalysis.analysis_name] = set(live)
        return live

    def _visit_decnum(self, e, ctx) -> _LiveSet:
        e.attribs[LiveVarAnalysis.analysis_name] = set()
        return set()

    def _visit_hexnum(self, e, ctx) -> _LiveSet:
        e.attribs[LiveVarAnalysis.analysis_name] = set()
        return set()

    def _visit_integer(self, e, ctx) -> _LiveSet:
        e.attribs[LiveVarAnalysis.analysis_name] = set()
        return set()

    def _visit_rational(self, e, ctx) -> _LiveSet:
        e.attribs[LiveVarAnalysis.analysis_name] = set()
        return set()

    def _visit_digits(self, e, ctx) -> _LiveSet:
        e.attribs[LiveVarAnalysis.analysis_name] = set()
        return set()

    def _visit_constant(self, e, ctx) -> _LiveSet:
        e.attribs[LiveVarAnalysis.analysis_name] = set()
        return set()

    def _visit_unaryop(self, e, ctx) -> _LiveSet:
        live = self._visit_expr(e.arg, ctx)
        e.attribs[LiveVarAnalysis.analysis_name] = set(live)
        return live

    def _visit_binaryop(self, e, ctx) -> _LiveSet:
        live = self._visit_expr(e.left, ctx) | self._visit_expr(e.right, ctx)
        e.attribs[LiveVarAnalysis.analysis_name] = set(live)
        return live

    def _visit_ternaryop(self, e, ctx) -> _LiveSet:
        live0 = self._visit_expr(e.arg0, ctx)
        live1 = self._visit_expr(e.arg1, ctx)
        live2 = self._visit_expr(e.arg2, ctx)
        live = live0 | live1 | live2
        e.attribs[LiveVarAnalysis.analysis_name] = set(live)
        return live

    def _visit_naryop(self, e, ctx) -> _LiveSet:
        live: set[str] = set()
        for arg in e.args:
            live |= self._visit_expr(arg, ctx)
        e.attribs[LiveVarAnalysis.analysis_name] = set(live)
        return live

    def _visit_compare(self, e, ctx) -> _LiveSet:
        live: set[str] = set()
        for arg in e.args:
            live |= self._visit_expr(arg, ctx)
        e.attribs[LiveVarAnalysis.analysis_name] = set(live)
        return live

    def _visit_call(self, e, ctx) -> _LiveSet:
        live: set[str] = set()
        for arg in e.args:
            live |= self._visit_expr(arg, ctx)
        e.attribs[LiveVarAnalysis.analysis_name] = set(live)
        return live

    def _visit_tuple_expr(self, e, ctx) -> _LiveSet:
        live: set[str] = set()
        for arg in e.args:
            live |= self._visit_expr(arg, ctx)
        e.attribs[LiveVarAnalysis.analysis_name] = set(live)
        return live

    def _visit_comp_expr(self, e, ctx) -> _LiveSet:
        live = self._visit_expr(e.elt, ctx)
        live -= set(e.vars)
        for iterable in e.iterables:
            live |= self._visit_expr(iterable, ctx)
        return live

    def _visit_ref_expr(self, e, ctx):
        live = self._visit(e.value, ctx)
        for s in e.slices:
            live |= self._visit(s, ctx)
        e.attribs[LiveVarAnalysis.analysis_name] = set(live)
        return live

    def _visit_if_expr(self, e, ctx) -> _LiveSet:
        cond_live = self._visit_expr(e.cond, ctx)
        ift_live = self._visit_expr(e.ift, ctx)
        iff_live = self._visit_expr(e.iff, ctx)
        live = cond_live | ift_live | iff_live
        e.attribs[LiveVarAnalysis.analysis_name] = set(live)
        return live

    def _visit_var_assign(self, stmt, ctx: _LiveSet) -> _LiveSet:
        ctx -= {stmt.var}
        return ctx | self._visit_expr(stmt.expr, None)

    def _visit_tuple_assign(self, stmt, ctx: _LiveSet) -> _LiveSet:
        ctx -= stmt.binding.names()
        return ctx | self._visit_expr(stmt.expr, None)

    def _visit_ref_assign(self, stmt, ctx: _LiveSet) -> _LiveSet:
        ctx |= self._visit_expr(stmt.expr, None)
        for s in stmt.slices:
            ctx |= self._visit_expr(s, None)
        ctx.add(stmt.var)
        return ctx

    def _visit_if_stmt(self, stmt, ctx: _LiveSet) -> _LiveSet:
        if stmt.iff is None:
            ctx |= self._visit_block(stmt.ift, set(ctx))
        else:
            ift_ctx = self._visit_block(stmt.ift, set(ctx))
            iff_ctx = self._visit_block(stmt.iff, set(ctx))
            ctx = ift_ctx | iff_ctx
        return ctx | self._visit_expr(stmt.cond, None)

    def _visit_while_stmt(self, stmt, ctx: _LiveSet) -> _LiveSet:
        ctx |= self._visit_block(stmt.body, set(ctx))
        return ctx | self._visit_expr(stmt.cond, None)

    def _visit_for_stmt(self, stmt, ctx: _LiveSet) -> _LiveSet:
        ctx |= self._visit_block(stmt.body, set(ctx))
        ctx -= {stmt.var}
        return ctx | self._visit_expr(stmt.iterable, None)

    def _visit_context(self, stmt, ctx: _LiveSet) -> _LiveSet:
        ctx = self._visit_block(stmt.body, set(ctx))
        if stmt.name is not None:
            ctx -= {stmt.name}
        return ctx

    def _visit_return(self, stmt, ctx: _LiveSet) -> _LiveSet:
        return self._visit_expr(stmt.expr, None)

    def _visit_block(self, block, ctx: _LiveSet) -> _LiveSet:
        block_out = set(ctx)
        for i, stmt in enumerate(reversed(block.stmts)):
            if isinstance(stmt, Return):
                assert i == 0, 'return statement must be the last statement'
                stmt_out = set() # override incoming out set
            else:
                stmt_out = set(ctx)
            ctx = self._visit_statement(stmt, ctx)
            stmt.attribs[LiveVarAnalysis.analysis_name] = (set(ctx), stmt_out)

        block.attribs[LiveVarAnalysis.analysis_name] = (set(ctx), block_out)
        return ctx

    def _visit_function(self, func, ctx: _LiveSet):
        return self._visit_block(func.body, ctx)

    # override for typing hint
    def _visit_expr(self, e: Expr, ctx: None) -> _LiveSet:
        return super()._visit_expr(e, ctx)
    
    # override for typing hint
    def _visit_statement(self, stmt: Stmt, ctx: _LiveSet) -> _LiveSet:
        return super()._visit_statement(stmt, ctx)


class LiveVarAnalysis:
    """Live variable analysis for the FPy AST."""

    analysis_name = 'live_vars'

    @staticmethod
    def analyze(func: FunctionDef):
        """Analyze the live variables in a function."""
        if not isinstance(func, FunctionDef):
            raise TypeError(f'expected a Function, got {func}')
        LiveVarAnalysisInstance().analyze(func)
