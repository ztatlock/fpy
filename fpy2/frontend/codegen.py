"""
This module does intermediate code generation, compiling
the abstract syntax tree (AST) to the intermediate representation (IR).
"""

from .definition import DefinitionAnalysis
from .fpyast import *
from .live_vars import LiveVarAnalysis
from .visitor import AstVisitor

from .. import ir
from ..utils import Gensym

class _IRCodegenInstance(AstVisitor):
    """Single-use instance of lowering an AST to an IR."""
    func: FunctionDef

    def __init__(self, func: FunctionDef):
        self.func = func

    def lower(self) -> ir.FunctionDef:
        return self._visit_function(self.func, None)

    def _visit_var(self, e, ctx: None):
        return ir.Var(e.name)

    def _visit_decnum(self, e, ctx: None):
        return ir.Decnum(e.val)

    def _visit_hexnum(self, e, ctx: None):
        return ir.Hexnum(e.val)

    def _visit_integer(self, e, ctx: None):
        return ir.Integer(e.val)

    def _visit_rational(self, e, ctx: None):
        return ir.Rational(e.p, e.q)

    def _visit_digits(self, e, ctx: None):
        return ir.Digits(e.m, e.e, e.b)

    def _visit_constant(self, e, ctx: None):
        return ir.Constant(e.val)

    # TODO: refactor into table lookup
    def _visit_unaryop(self, e, ctx: None):
        match e.op:
            case UnaryOpKind.NEG:
                arg = self._visit_expr(e.arg, ctx)
                return ir.Neg(arg)
            case UnaryOpKind.NOT:
                arg = self._visit_expr(e.arg, ctx)
                return ir.Not(arg)
            case UnaryOpKind.FABS:
                arg = self._visit_expr(e.arg, ctx)
                return ir.Fabs(arg)
            case UnaryOpKind.SQRT:
                arg = self._visit_expr(e.arg, ctx)
                return ir.Sqrt(arg)
            case UnaryOpKind.CBRT:
                arg = self._visit_expr(e.arg, ctx)
                return ir.Cbrt(arg)
            case UnaryOpKind.CEIL:
                arg = self._visit_expr(e.arg, ctx)
                return ir.Ceil(arg)
            case UnaryOpKind.FLOOR:
                arg = self._visit_expr(e.arg, ctx)
                return ir.Floor(arg)
            case UnaryOpKind.NEARBYINT:
                arg = self._visit_expr(e.arg, ctx)
                return ir.Nearbyint(arg)
            case UnaryOpKind.ROUND:
                arg = self._visit_expr(e.arg, ctx)
                return ir.Round(arg)
            case UnaryOpKind.TRUNC:
                arg = self._visit_expr(e.arg, ctx)
                return ir.Trunc(arg)
            case UnaryOpKind.ACOS:
                arg = self._visit_expr(e.arg, ctx)
                return ir.Acos(arg)
            case UnaryOpKind.ASIN:
                arg = self._visit_expr(e.arg, ctx)
                return ir.Asin(arg)
            case UnaryOpKind.ATAN:
                arg = self._visit_expr(e.arg, ctx)
                return ir.Atan(arg)
            case UnaryOpKind.COS:
                arg = self._visit_expr(e.arg, ctx)
                return ir.Cos(arg)
            case UnaryOpKind.SIN:
                arg = self._visit_expr(e.arg, ctx)
                return ir.Sin(arg)
            case UnaryOpKind.TAN:
                arg = self._visit_expr(e.arg, ctx)
                return ir.Tan(arg)
            case UnaryOpKind.ACOSH:
                arg = self._visit_expr(e.arg, ctx)
                return ir.Acosh(arg)
            case UnaryOpKind.ASINH:
                arg = self._visit_expr(e.arg, ctx)
                return ir.Asinh(arg)
            case UnaryOpKind.ATANH:
                arg = self._visit_expr(e.arg, ctx)
                return ir.Atanh(arg)
            case UnaryOpKind.COSH:
                arg = self._visit_expr(e.arg, ctx)
                return ir.Cosh(arg)
            case UnaryOpKind.SINH:
                arg = self._visit_expr(e.arg, ctx)
                return ir.Sinh(arg)
            case UnaryOpKind.TANH:
                arg = self._visit_expr(e.arg, ctx)
                return ir.Tanh(arg)
            case UnaryOpKind.EXP:
                arg = self._visit_expr(e.arg, ctx)
                return ir.Exp(arg)
            case UnaryOpKind.EXP2:
                arg = self._visit_expr(e.arg, ctx)
                return ir.Exp2(arg)
            case UnaryOpKind.EXPM1:
                arg = self._visit_expr(e.arg, ctx)
                return ir.Expm1(arg)
            case UnaryOpKind.LOG:
                arg = self._visit_expr(e.arg, ctx)
                return ir.Log(arg)
            case UnaryOpKind.LOG10:
                arg = self._visit_expr(e.arg, ctx)
                return ir.Log10(arg)
            case UnaryOpKind.LOG1P:
                arg = self._visit_expr(e.arg, ctx)
                return ir.Log1p(arg)
            case UnaryOpKind.LOG2:
                arg = self._visit_expr(e.arg, ctx)
                return ir.Log2(arg)
            case UnaryOpKind.ERF:
                arg = self._visit_expr(e.arg, ctx)
                return ir.Erf(arg)
            case UnaryOpKind.ERFC:
                arg = self._visit_expr(e.arg, ctx)
                return ir.Erfc(arg)
            case UnaryOpKind.LGAMMA:
                arg = self._visit_expr(e.arg, ctx)
                return ir.Lgamma(arg)
            case UnaryOpKind.TGAMMA:
                arg = self._visit_expr(e.arg, ctx)
                return ir.Tgamma(arg)
            case UnaryOpKind.ISFINITE:
                arg = self._visit_expr(e.arg, ctx)
                return ir.IsFinite(arg)
            case UnaryOpKind.ISINF:
                arg = self._visit_expr(e.arg, ctx)
                return ir.IsInf(arg)
            case UnaryOpKind.ISNAN:
                arg = self._visit_expr(e.arg, ctx)
                return ir.IsNan(arg)
            case UnaryOpKind.ISNORMAL:
                arg = self._visit_expr(e.arg, ctx)
                return ir.IsNormal(arg)
            case UnaryOpKind.SIGNBIT:
                arg = self._visit_expr(e.arg, ctx)
                return ir.Signbit(arg)
            case UnaryOpKind.CAST:
                arg = self._visit_expr(e.arg, ctx)
                return ir.Cast(arg)
            case UnaryOpKind.RANGE:
                arg = self._visit_expr(e.arg, ctx)
                return ir.Range(arg)
            case UnaryOpKind.DIM:
                arg = self._visit_expr(e.arg, ctx)
                return ir.Dim(arg)
            case _:
                raise NotImplementedError('unexpected op', e.op)

    def _visit_binaryop(self, e, ctx: None):
        lhs = self._visit_expr(e.left, ctx)
        rhs = self._visit_expr(e.right, ctx)
        match e.op:
            case BinaryOpKind.ADD:
                return ir.Add(lhs, rhs)
            case BinaryOpKind.SUB:
                return ir.Sub(lhs, rhs)
            case BinaryOpKind.MUL:
                return ir.Mul(lhs, rhs)
            case BinaryOpKind.DIV:
                return ir.Div(lhs, rhs)
            case BinaryOpKind.COPYSIGN:
                return ir.Copysign(lhs, rhs)
            case BinaryOpKind.FDIM:
                return ir.Fdim(lhs, rhs)
            case BinaryOpKind.FMAX:
                return ir.Fmax(lhs, rhs)
            case BinaryOpKind.FMIN:
                return ir.Fmin(lhs, rhs)
            case BinaryOpKind.FMOD:
                return ir.Fmod(lhs, rhs)
            case BinaryOpKind.REMAINDER:
                return ir.Remainder(lhs, rhs)
            case BinaryOpKind.HYPOT:
                return ir.Hypot(lhs, rhs)
            case BinaryOpKind.ATAN2:
                return ir.Atan2(lhs, rhs)
            case BinaryOpKind.POW:
                return ir.Pow(lhs, rhs)
            case BinaryOpKind.SIZE:
                return ir.Size(lhs, rhs)
            case _:
                raise NotImplementedError('unexpected op', e.op)

    def _visit_ternaryop(self, e, ctx: None):
        arg0 = self._visit_expr(e.arg0, ctx)
        arg1 = self._visit_expr(e.arg1, ctx)
        arg2 = self._visit_expr(e.arg2, ctx)
        match e.op:
            case TernaryOpKind.FMA:
                return ir.Fma(arg0, arg1, arg2)
            case _:
                raise NotImplementedError('unexpected op', e.op)

    def _visit_naryop(self, e, ctx: None):
        args = [self._visit_expr(arg, ctx) for arg in e.args]
        match e.op:
            case NaryOpKind.AND:
                return ir.And(*args)
            case NaryOpKind.OR:
                return ir.Or(*args)
            case _:
                raise NotImplementedError('unexpected op', e.op)

    def _visit_compare(self, e, ctx: None):
        args = [self._visit_expr(arg, ctx) for arg in e.args]
        return ir.Compare(e.ops, args)

    def _visit_call(self, e, ctx: None):
        args = [self._visit_expr(arg, ctx) for arg in e.args]
        return ir.UnknownCall(e.op, *args)

    def _visit_tuple_expr(self, e, ctx: None):
        elts = [self._visit_expr(arg, ctx) for arg in e.args]
        return ir.TupleExpr(*elts)

    def _visit_comp_expr(self, e, ctx: None):
        iterables = [self._visit_expr(arg, ctx) for arg in e.iterables]
        elt = self._visit_expr(e.elt, ctx)
        return ir.CompExpr(e.vars, iterables, elt)

    def _visit_ref_expr(self, e, ctx: None):
        value = self._visit_expr(e.value, ctx)
        slices = [self._visit_expr(s, ctx) for s in e.slices]
        return ir.TupleRef(value, *slices)

    def _visit_if_expr(self, e, ctx: None):
        cond = self._visit_expr(e.cond, ctx)
        ift = self._visit_expr(e.ift, ctx)
        iff = self._visit_expr(e.iff, ctx)
        return ir.IfExpr(cond, ift, iff)

    def _visit_var_assign(self, stmt, ctx: None):
        expr = self._visit_expr(stmt.expr, ctx)
        return ir.VarAssign(stmt.var, ir.AnyType(), expr)

    def _visit_tuple_binding(self, vars: TupleBinding):
        new_vars: list[str | ir.TupleBinding] = []
        for name in vars:
            if isinstance(name, str):
                new_vars.append(name)
            elif isinstance(name, TupleBinding):
                new_vars.append(self._visit_tuple_binding(name))
            else:
                raise NotImplementedError('unexpected tuple identifier', name)
        return ir.TupleBinding(new_vars)

    def _visit_tuple_assign(self, stmt, ctx: None):
        binding = self._visit_tuple_binding(stmt.binding)
        expr = self._visit_expr(stmt.expr, ctx)
        return ir.TupleAssign(binding, ir.AnyType(), expr)

    def _visit_ref_assign(self, stmt, ctx: None):
        slices = [self._visit_expr(s, ctx) for s in stmt.slices]
        value = self._visit_expr(stmt.expr, ctx)
        return ir.RefAssign(stmt.var, slices, value)

    def _visit_if_stmt(self, stmt, ctx: None):
        cond = self._visit_expr(stmt.cond, ctx)
        ift = self._visit_block(stmt.ift, ctx)
        if stmt.iff is None:
            return ir.If1Stmt(cond, ift, [])
        else:
            iff = self._visit_block(stmt.iff, ctx)
            return ir.IfStmt(cond, ift, iff, [])

    def _visit_while_stmt(self, stmt, ctx: None):
        cond = self._visit_expr(stmt.cond, ctx)
        body = self._visit_block(stmt.body, ctx)
        return ir.WhileStmt(cond, body, [])

    def _visit_for_stmt(self, stmt, ctx: None):
        iterable = self._visit_expr(stmt.iterable, ctx)
        body = self._visit_block(stmt.body, ctx)
        return ir.ForStmt(stmt.var, ir.AnyType(), iterable, body, [])

    def _visit_context(self, stmt, ctx: None):
        block = self._visit_block(stmt.body, ctx)
        return ir.ContextStmt(stmt.name, stmt.props, block)

    def _visit_return(self, stmt, ctx: None):
        return ir.Return(self._visit_expr(stmt.expr, ctx))

    def _visit_block(self, block, ctx: None):
        return([self._visit_statement(stmt, ctx) for stmt in block.stmts])

    def _visit_function(self, func, ctx: None):
        args: list[ir.Argument] = []
        for arg in func.args:
            # TODO: use type annotation
            ty = ir.AnyType()
            args.append(ir.Argument(arg.name, ty))
        e = self._visit_block(func.body, ctx)
        return ir.FunctionDef(func.name, args, e, ir.AnyType(), func.ctx)

    # override for typing hint
    def _visit_expr(self, e: Expr, ctx: None) -> ir.Expr:
        return super()._visit_expr(e, ctx)

    # override for typing hint
    def _visit_statement(self, stmt: Stmt, ctx: None) -> ir.Stmt:
        return super()._visit_statement(stmt, ctx)


class IRCodegen:
    """Lowers a FPy AST to FPy IR."""

    @staticmethod
    def lower(f: FunctionDef) -> ir.FunctionDef:
        return _IRCodegenInstance(f).lower()
