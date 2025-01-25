"""
This module implements a reaching definitions analysis pass for the IR.
"""

from dataclasses import dataclass

from ..ir import *

@dataclass
class Reach:
    reach_in: set[str]
    reach_out: set[str]
    kill_out: set[str]

_BlockCtx = set[str]
_StmtCtx = tuple[set[str], set[str]]
_RetType = tuple[set[str], set[str]]

class _ReachingDefsInstance(DefaultVisitor):
    """Single-use instance of the reaching definitions analysis."""
    func: FunctionDef
    reaches: dict[Block, Reach]

    def __init__(self, func: FunctionDef):
        self.func = func
        self.reaches = {}

    def analyze(self):
        self._visit(self.func, None)
        return self.reaches

    def _visit_var_assign(self, stmt, ctx: _StmtCtx) -> _RetType:
        defs_in, kill_in = ctx
        defs_out = { *defs_in, stmt.var }
        kill_out = kill_in | (defs_in & { stmt.var })
        return defs_out, kill_out

    def _visit_tuple_assign(self, stmt, ctx: _StmtCtx) -> _RetType:
        defs_in, kill_in = ctx
        gen = set(stmt.binding.names())
        defs_out = defs_in | gen
        kill_out = kill_in | (defs_in & gen)
        return defs_out, kill_out

    def _visit_ref_assign(self, stmt, ctx: _StmtCtx) -> _RetType:
        return ctx

    def _visit_if1_stmt(self, stmt, ctx: _StmtCtx) -> _RetType:
        defs_in, kill_in = ctx
        _, block_kill = self._visit_block(stmt.body, defs_in.copy())

        defs_out = defs_in.copy()
        kill_out = kill_in | (defs_in & block_kill)
        return defs_out, kill_out

    def _visit_if_stmt(self, stmt, ctx: _StmtCtx) -> _RetType:
        defs_in, kill_in = ctx
        ift_defs, ift_kill = self._visit_block(stmt.ift, defs_in.copy())
        iff_defs, iff_kill = self._visit_block(stmt.iff, defs_in.copy())

        defs_out = ift_defs & iff_defs
        kill_out = kill_in | (defs_in & (ift_kill | iff_kill))
        return defs_out, kill_out

    def _visit_while_stmt(self, stmt, ctx: _StmtCtx) -> _RetType:
        defs_in, kill_in = ctx
        _, block_kill = self._visit_block(stmt.body, defs_in.copy())

        defs_out = defs_in.copy()
        kill_out = kill_in | (defs_in & block_kill)
        return defs_out, kill_out

    def _visit_for_stmt(self, stmt, ctx: _StmtCtx) -> _RetType:
        defs_in, kill_in = ctx
        defs = defs_in | { stmt.var }
        _, block_kill = self._visit_block(stmt.body, defs.copy())

        defs_out = defs.copy()
        kill_out = kill_in | (defs & block_kill)
        return defs_out, kill_out

    def _visit_context(self, stmt, ctx: _StmtCtx) -> _RetType:
        # TODO: handle `stmt.name`
        defs_in, kill_in = ctx
        _, block_kill = self._visit_block(stmt.body, defs_in.copy())

        defs_out = defs_in.copy()
        kill_out = kill_in | (defs_in & block_kill)
        return defs_out, kill_out

    def _visit_return(self, stmt, ctx: _StmtCtx) -> _RetType:
        return ctx

    def _visit_block(self, block, ctx: _BlockCtx):
        reach_in = ctx.copy()

        defs: set[str] = reach_in.copy()
        kill: set[str] = set()
        for stmt in block.stmts:
            defs, kill = self._visit(stmt, (defs, kill))

        reach_out = defs | (reach_in - kill)
        self.reaches[block] = Reach(reach_in, reach_out, kill)
        return defs, kill

    def _visit_function(self, func: FunctionDef, _):
        ctx: set[str] = set()
        for arg in func.args:
            ctx.add(arg.name)
        self._visit_block(func.body, ctx)

    # override to get typing hint
    def _visit_statement(self, stmt, ctx: _StmtCtx) -> _RetType:
        return super()._visit_statement(stmt, ctx)

class ReachingDefs:
    """
    Reaching definitions analysis for the FPy IR.
    """

    analysis_name = 'reaching_defs'

    @staticmethod
    def analyze(func: FunctionDef) -> dict[Block, Reach]:
        return _ReachingDefsInstance(func).analyze()
