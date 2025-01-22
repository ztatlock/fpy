"""
FPy parsing from FPCore.
"""

import titanfp.fpbench.fpcast as fpc

from .codegen import IRCodegen
from .definition import DefinitionAnalysis
from .fpyast import *
from .live_vars import LiveVarAnalysis
from .syntax_check import SyntaxCheck

from ..passes import VerifyIR

from ..utils import Gensym

class _Ctx:
    env: dict[str, str]
    stmts: list[Stmt]

    def __init__(self):
        self.env = {}
        self.stmts = []

    def without_stmts(self):
        ctx = _Ctx()
        ctx.env = dict(self.env)
        return ctx


class _FPCore2FPy:
    core: fpc.FPCore
    gensym: Gensym

    def __init__(self, core: fpc.FPCore):
        self.core = core
        self.gensym = Gensym()

    def _visit_decnum(self, e: fpc.Decnum, ctx: _Ctx) -> Expr:
        return Decnum(str(e.value), None)

    def _visit_hexnum(self, e: fpc.Hexnum, ctx: _Ctx) -> Expr:
        return Hexnum(str(e.value), None)

    def _visit_integer(self, e: fpc.Integer, ctx: _Ctx) -> Expr:
        return Integer(e.value, None)

    def _visit_rational(self, e: fpc.Rational, ctx: _Ctx) -> Expr:
        return Rational(e.p, e.q, None)

    def _visit_constant(self, e: fpc.Constant, ctx: _Ctx) -> Expr:
        return Constant(str(e.value), None)

    def _visit_nary(self, e: fpc.NaryExpr, ctx: _Ctx) -> Expr:
        match e:
            case fpc.And():
                exprs = [self._visit(e, ctx) for e in e.children]
                return NaryOp(NaryOpKind.AND, exprs, None)
            case fpc.Or():
                exprs = [self._visit(e, ctx) for e in e.children]
                return NaryOp(NaryOpKind.OR, exprs, None)
            case fpc.LT():
                assert len(e.children) >= 2, "not enough children"
                ops = [CompareOp.LT for _ in e.children[1:]]
                exprs = [self._visit(e, ctx) for e in e.children]
                return Compare(ops, exprs, None)
            case _:
                raise NotImplementedError('unexpected FPCore expression', e)

    def _visit_if_stmt(self, e: fpc.If, ctx: _Ctx) -> Expr:
        # create new blocks
        ift_ctx = ctx.without_stmts()
        iff_ctx = ctx.without_stmts()

        # compile children
        cond_expr = self._visit(e.cond, ctx)
        ift_expr = self._visit(e.then_body, ift_ctx)
        iff_expr = self._visit(e.else_body, iff_ctx)

        # emit temporary to bind result of branches
        t = self.gensym.fresh('t')
        ift_ctx.stmts.append(VarAssign(t, ift_expr, None, None))
        iff_ctx.stmts.append(VarAssign(t, iff_expr, None, None))

        # create if statement and bind it
        if_stmt = IfStmt(cond_expr, Block(ift_ctx.stmts), Block(iff_ctx.stmts), None)
        ctx.stmts.append(if_stmt)

        return Var(t, None)

    def _visit(self, e: fpc.Expr, ctx: _Ctx) -> Expr:
        match e:
            case fpc.Decnum():
                return self._visit_decnum(e, ctx)
            case fpc.Hexnum():
                return self._visit_hexnum(e, ctx)
            case fpc.Integer():
                return self._visit_integer(e, ctx)
            case fpc.Rational():
                return self._visit_rational(e, ctx)
            case fpc.Constant():
                return self._visit_constant(e, ctx)
            case fpc.NaryExpr():
                return self._visit_nary(e, ctx)
            case fpc.If():
                return self._visit_if_stmt(e, ctx)
            case _:
                raise NotImplementedError(f'cannot convert to FPy {e}')

    def _visit_function(self, f: fpc.FPCore):
        if f.inputs != []:
            raise NotImplementedError('args')
        # TODO: parse metadata
        props = dict(f.props)

        ctx = _Ctx()
        e = self._visit(f.e, ctx)
        block = Block(ctx.stmts + [Return(e, None)])

        ast = FunctionDef(f.ident, [], block, None)
        ast.ctx = props
        return ast

    def convert(self) -> FunctionDef:
        ast = self._visit_function(self.core)
        if not isinstance(ast, FunctionDef):
            raise TypeError(f'should have produced a FunctionDef {ast}')

        return ast

def fpcore_to_fpy(core: fpc.FPCore):
    ast = _FPCore2FPy(core).convert()
    
    # analyze and lower to the IR
    SyntaxCheck.analyze(ast)
    DefinitionAnalysis.analyze(ast)
    LiveVarAnalysis.analyze(ast)
    ir = IRCodegen.lower(ast)
    VerifyIR.check(ir)

    return ir
