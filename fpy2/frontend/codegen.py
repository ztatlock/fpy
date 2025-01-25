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
        return self._visit(self.func, None)

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

    def _visit_unaryop(self, e, ctx: None):
        match e.op:
            case UnaryOpKind.NEG:
                arg = self._visit(e.arg, ctx)
                return ir.Neg(arg)
            case UnaryOpKind.NOT:
                arg = self._visit(e.arg, ctx)
                return ir.Not(arg)
            case UnaryOpKind.FABS:
                arg = self._visit(e.arg, ctx)
                return ir.Fabs(arg)
            case UnaryOpKind.SQRT:
                arg = self._visit(e.arg, ctx)
                return ir.Sqrt(arg)
            case UnaryOpKind.CBRT:
                arg = self._visit(e.arg, ctx)
                return ir.Cbrt(arg)
            case UnaryOpKind.CEIL:
                arg = self._visit(e.arg, ctx)
                return ir.Ceil(arg)
            case UnaryOpKind.FLOOR:
                arg = self._visit(e.arg, ctx)
                return ir.Floor(arg)
            case UnaryOpKind.NEARBYINT:
                arg = self._visit(e.arg, ctx)
                return ir.Nearbyint(arg)
            case UnaryOpKind.ROUND:
                arg = self._visit(e.arg, ctx)
                return ir.Round(arg)
            case UnaryOpKind.TRUNC:
                arg = self._visit(e.arg, ctx)
                return ir.Trunc(arg)
            case UnaryOpKind.ACOS:
                arg = self._visit(e.arg, ctx)
                return ir.Acos(arg)
            case UnaryOpKind.ASIN:
                arg = self._visit(e.arg, ctx)
                return ir.Asin(arg)
            case UnaryOpKind.ATAN:
                arg = self._visit(e.arg, ctx)
                return ir.Atan(arg)
            case UnaryOpKind.COS:
                arg = self._visit(e.arg, ctx)
                return ir.Cos(arg)
            case UnaryOpKind.SIN:
                arg = self._visit(e.arg, ctx)
                return ir.Sin(arg)
            case UnaryOpKind.TAN:
                arg = self._visit(e.arg, ctx)
                return ir.Tan(arg)
            case UnaryOpKind.ACOSH:
                arg = self._visit(e.arg, ctx)
                return ir.Acosh(arg)
            case UnaryOpKind.ASINH:
                arg = self._visit(e.arg, ctx)
                return ir.Asinh(arg)
            case UnaryOpKind.ATANH:
                arg = self._visit(e.arg, ctx)
                return ir.Atanh(arg)
            case UnaryOpKind.COSH:
                arg = self._visit(e.arg, ctx)
                return ir.Cosh(arg)
            case UnaryOpKind.SINH:
                arg = self._visit(e.arg, ctx)
                return ir.Sinh(arg)
            case UnaryOpKind.TANH:
                arg = self._visit(e.arg, ctx)
                return ir.Tanh(arg)
            case UnaryOpKind.EXP:
                arg = self._visit(e.arg, ctx)
                return ir.Exp(arg)
            case UnaryOpKind.EXP2:
                arg = self._visit(e.arg, ctx)
                return ir.Exp2(arg)
            case UnaryOpKind.EXPM1:
                arg = self._visit(e.arg, ctx)
                return ir.Expm1(arg)
            case UnaryOpKind.LOG:
                arg = self._visit(e.arg, ctx)
                return ir.Log(arg)
            case UnaryOpKind.LOG10:
                arg = self._visit(e.arg, ctx)
                return ir.Log10(arg)
            case UnaryOpKind.LOG1P:
                arg = self._visit(e.arg, ctx)
                return ir.Log1p(arg)
            case UnaryOpKind.LOG2:
                arg = self._visit(e.arg, ctx)
                return ir.Log2(arg)
            case UnaryOpKind.ERF:
                arg = self._visit(e.arg, ctx)
                return ir.Erf(arg)
            case UnaryOpKind.ERFC:
                arg = self._visit(e.arg, ctx)
                return ir.Erfc(arg)
            case UnaryOpKind.LGAMMA:
                arg = self._visit(e.arg, ctx)
                return ir.Lgamma(arg)
            case UnaryOpKind.TGAMMA:
                arg = self._visit(e.arg, ctx)
                return ir.Tgamma(arg)
            case UnaryOpKind.ISFINITE:
                arg = self._visit(e.arg, ctx)
                return ir.IsFinite(arg)
            case UnaryOpKind.ISINF:
                arg = self._visit(e.arg, ctx)
                return ir.IsInf(arg)
            case UnaryOpKind.ISNAN:
                arg = self._visit(e.arg, ctx)
                return ir.IsNan(arg)
            case UnaryOpKind.ISNORMAL:
                arg = self._visit(e.arg, ctx)
                return ir.IsNormal(arg)
            case UnaryOpKind.SIGNBIT:
                arg = self._visit(e.arg, ctx)
                return ir.Signbit(arg)
            case UnaryOpKind.CAST:
                arg = self._visit(e.arg, ctx)
                return ir.Cast(arg)
            case UnaryOpKind.RANGE:
                arg = self._visit(e.arg, ctx)
                return ir.Range(arg)
            case _:
                raise NotImplementedError('unexpected op', e.op)

    def _visit_binaryop(self, e, ctx: None):
        lhs = self._visit(e.left, ctx)
        rhs = self._visit(e.right, ctx)
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
        arg0 = self._visit(e.arg0, ctx)
        arg1 = self._visit(e.arg1, ctx)
        arg2 = self._visit(e.arg2, ctx)
        match e.op:
            case TernaryOpKind.FMA:
                return ir.Fma(arg0, arg1, arg2)
            case _:
                raise NotImplementedError('unexpected op', e.op)

    def _visit_naryop(self, e, ctx: None):
        args: list[ir.Expr] = [self._visit(arg, ctx) for arg in e.args]
        match e.op:
            case NaryOpKind.AND:
                return ir.And(*args)
            case NaryOpKind.OR:
                return ir.Or(*args)
            case _:
                raise NotImplementedError('unexpected op', e.op)

    def _visit_compare(self, e, ctx: None):
        args: list[ir.Expr] = [self._visit(arg, ctx) for arg in e.args]
        return ir.Compare(e.ops, args)

    def _visit_call(self, e, ctx: None):
        args: list[ir.Expr]  = [self._visit(arg, ctx) for arg in e.args]
        return ir.UnknownCall(e.op, *args)

    def _visit_tuple_expr(self, e, ctx: None):
        elts = [self._visit(arg, ctx) for arg in e.args]
        return ir.TupleExpr(*elts)

    def _visit_comp_expr(self, e, ctx: None):
        iterables = [self._visit(arg, ctx) for arg in e.iterables]
        elt = self._visit(e.elt, ctx)
        return ir.CompExpr(e.vars, iterables, elt)

    def _visit_ref_expr(self, e, ctx: None):
        value = self._visit(e.value, ctx)
        slices = [self._visit(s, ctx) for s in e.slices]
        return ir.TupleRef(value, *slices)

    def _visit_if_expr(self, e, ctx: None):
        cond = self._visit(e.cond, ctx)
        ift = self._visit(e.ift, ctx)
        iff = self._visit(e.iff, ctx)
        return ir.IfExpr(cond, ift, iff)

    def _visit_var_assign(self, stmt, ctx: None):
        expr = self._visit(stmt.expr, ctx)
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
        expr = self._visit(stmt.expr, ctx)
        return ir.TupleAssign(binding, ir.AnyType(), expr)

    def _visit_ref_assign(self, stmt, ctx: None):
        slices = [self._visit(s, ctx) for s in stmt.slices]
        value = self._visit(stmt.expr, ctx)
        return ir.RefAssign(stmt.var, slices, value)

    def _visit_if_stmt(self, stmt, ctx: None):
        cond = self._visit(stmt.cond, ctx)
        ift = self._visit(stmt.ift, ctx)
        if stmt.iff is None:
            return ir.If1Stmt(cond, ift, [])
        else:
            iff = self._visit(stmt.iff, ctx)
            return ir.IfStmt(cond, ift, iff, [])

    def _visit_while_stmt(self, stmt, ctx: None):
        cond = self._visit(stmt.cond, ctx)
        body = self._visit(stmt.body, ctx)
        return ir.WhileStmt(cond, body, [])

    def _visit_for_stmt(self, stmt, ctx: None):
        iterable = self._visit(stmt.iterable, ctx)
        body = self._visit(stmt.body, ctx)
        return ir.ForStmt(stmt.var, ir.AnyType(), iterable, body, [])

    def _visit_context(self, stmt, ctx: None):
        block = self._visit(stmt.body, ctx)
        return ir.ContextStmt(stmt.name, stmt.props, block)

    def _visit_return(self, stmt, ctx: None):
        return ir.Return(self._visit(stmt.expr, ctx))

    def _visit_block(self, block, ctx: None):
        return ir.Block([self._visit(stmt, ctx) for stmt in block.stmts])

    def _visit_function(self, func, ctx: None):
        args: list[ir.Argument] = []
        for arg in func.args:
            # TODO: use type annotation
            ty = ir.AnyType()
            args.append(ir.Argument(arg.name, ty))
        e = self._visit(func.body, ctx)
        return ir.FunctionDef(func.name, args, e, ir.AnyType(), func.ctx)




# class _IRCodegenInstance(AstVisitor):
#     """Instance of lowering an AST to an IR."""
#     func: FunctionDef
#     gensym: Gensym

#     def __init__(self, func: FunctionDef):
#         self.func = func
#         self.gensym = Gensym()

#     def lower(self) -> ir.FunctionDef:
#         return self._visit(self.func, {})



#     def _visit_comp_expr(self, e, ctx: None):
#         iterables = [self._visit(arg, ctx: None) for arg in e.iterables]
#         # generate fresh variable for the loop variable
#         iter_vars: list[str] = []
#         for var in e.vars:
#             iter_var = self.gensym.fresh(var)
#             ctx = { **ctx, var: iter_var }
#             iter_vars.append(iter_var)
#         # compile the loop body
#         elt = self._visit(e.elt, ctx: None)
#         return ir.CompExpr(iter_vars, iterables, elt)

#     def _visit_ref_expr(self, e, ctx: None):
#         value = self._visit(e.value, ctx: None)
#         slices = [self._visit(s, ctx: None) for s in e.slices]
#         return ir.TupleRef(value, *slices)

#     def _visit_if_expr(self, e, ctx: None):
#         return ir.IfExpr(
#             self._visit(e.cond, ctx: None),
#             self._visit(e.ift, ctx: None),
#             self._visit(e.iff, ctx: None)
#         )

#     def _visit_var_assign(self, stmt, ctx: None):
#         # compile the expression
#         e = self._visit(stmt.expr, ctx: None)
#         # generate fresh variable for the assignment
#         t = self.gensym.fresh(stmt.var)
#         ctx = { **ctx, stmt.var: t }
#         s = ir.VarAssign(t, ir.AnyType(), e)
#         return s, ctx
    
#     def _compile_tuple_binding(self, vars: TupleBinding, ctx: None):
#         new_vars: list[str | ir.TupleBinding] = []
#         for name in vars:
#             if isinstance(name, str):
#                 new_vars.append(ctx[name])
#             elif isinstance(name, TupleBinding):
#                 new_vars.append(self._compile_tuple_binding(name, ctx: None))
#             else:
#                 raise NotImplementedError('unexpected tuple identifier', name)
#         return ir.TupleBinding(new_vars)

#     def _visit_tuple_assign(self, stmt, ctx: None):
#         # compile the expression
#         e = self._visit(stmt.expr, ctx: None)
#         # generate fresh variables for the tuple assignment
#         for name in stmt.binding.names():
#             t = self.gensym.fresh(name)
#             ctx = { **ctx, name: t }
#         vars = self._compile_tuple_binding(stmt.binding, ctx: None)
#         tys = ir.TensorType([ir.AnyType() for _ in stmt.binding])
#         s = ir.TupleAssign(vars, tys, e)
#         return s, ctx

#     def _visit_ref_assign(self, stmt, ctx: None):
#         slices = [self._visit(s, ctx: None) for s in stmt.slices]
#         e = self._visit(stmt.expr, ctx: None)
#         s = ir.RefAssign(stmt.var, slices, e)
#         return s, ctx


    # def _visit_if1_stmt(self, stmt: IfStmt, ctx: None):
    #     """Like `_visit_if_stmt`, but for 1-armed if statements."""
    #     cond = self._visit(stmt.cond, ctx: None)
    #     body, body_ctx = self._visit_block(stmt.ift, ctx: None)
    #     _, live_out = stmt.attribs[LiveVarAnalysis.analysis_name]
    #     # merge live variables and create new context
    #     phis: list[ir.PhiNode] = []
    #     new_ctx: None = dict()
    #     for name in live_out:
    #         old_name = ctx[name]
    #         new_name = body_ctx[name]
    #         if old_name != new_name:
    #             t = self.gensym.fresh(name)
    #             phis.append(ir.PhiNode(t, old_name, new_name, ir.AnyType()))
    #             new_ctx[name] = t
    #         else:
    #             new_ctx[name] = ctx[name]
    #     # create new statement
    #     s = ir.If1Stmt(cond, body, phis)
    #     return s, new_ctx
    
    # def _visit_if2_stmt(self, stmt: IfStmt, ctx: None):
    #     """Like `_visit_if_stmt`, but for 2-armed if statements."""
    #     assert stmt.iff is not None, 'expected a 2-armed if statement'
    #     cond = self._visit(stmt.cond, ctx: None)
    #     ift, ift_ctx = self._visit_block(stmt.ift, ctx: None)
    #     iff, iff_ctx = self._visit_block(stmt.iff, ctx: None)
    #     _, live_out = stmt.attribs[LiveVarAnalysis.analysis_name]
    #     # merge live variables
    #     phis: list[ir.PhiNode] = []
    #     new_ctx: None = dict()
    #     for name in live_out:
    #         # well-formedness means that the variable is in both contexts
    #         ift_name = ift_ctx.get(name, None)
    #         iff_name = iff_ctx.get(name, None)
    #         assert ift_name is not None, f'variable not in true branch {ift_name}'
    #         assert iff_name is not None, f'variable not in false branch {iff_name}'
    #         if ift_name != iff_name:
    #             # variable updated on at least one branch => create phi node
    #             t = self.gensym.fresh(name)
    #             phis.append(ir.PhiNode(t, ift_name, iff_name, ir.AnyType()))
    #             new_ctx[name] = t
    #         else:
    #             # variable not mutated => keep the same name
    #             new_ctx[name] = ctx[name]
    #     # create new statement
    #     s = ir.IfStmt(cond, ift, iff, phis)
    #     return s, new_ctx

    # def _visit_if_stmt(self, stmt, ctx: None):
    #     if stmt.iff is None:
    #         return self._visit_if1_stmt(stmt, ctx: None)
    #     else:
    #         return self._visit_if2_stmt(stmt, ctx: None)

    # def _visit_while_stmt(self, stmt, ctx: None):
    #     # merge variables initialized before the block that
    #     # are updated in the body of the loop
    #     live_in, _ = stmt.attribs[LiveVarAnalysis.analysis_name]
    #     _, def_out = stmt.body.attribs[DefinitionAnalysis.analysis_name]
    #     # generate fresh variables for all changed variables
    #     changed_map: dict[str, str] = dict()
    #     changed_vars: set[str] = live_in & def_out
    #     for name in changed_vars:
    #         t = self.gensym.fresh(name)
    #         changed_map[name] = t
    #     # create the new context for the loop
    #     loop_ctx: None = dict()
    #     for name in ctx:
    #         if name in changed_map:
    #             loop_ctx[name] = changed_map[name]
    #         else:
    #             loop_ctx[name] = ctx[name]
    #     # compile the condition and body using the loop context
    #     cond = self._visit(stmt.cond, loop_ctx: None)
    #     body, body_ctx = self._visit_block(stmt.body, loop_ctx: None)
    #     # merge all changed variables using phi nodes
    #     phis: list[ir.PhiNode] = []
    #     for name, t in changed_map.items():
    #         old_name = ctx[name]
    #         new_name = body_ctx[name]
    #         assert old_name != new_name, 'must be different by definition analysis'
    #         phis.append(ir.PhiNode(t, old_name, new_name, ir.AnyType()))
    #     # create new statement and context
    #     s = ir.WhileStmt(cond, body, phis)
    #     new_ctx: None = dict()
    #     for name in ctx:
    #         if name in changed_map:
    #             new_ctx[name] = changed_map[name]
    #         else:
    #             new_ctx[name] = ctx[name]
    #     return s, new_ctx

    # def _visit_for_stmt(self, stmt, ctx: None):
    #     # compile the iterable expression
    #     cond = self._visit(stmt.iterable, ctx: None)
    #     # generate fresh variable for the loop variable
    #     iter_var = self.gensym.fresh(stmt.var)
    #     ctx = { **ctx, stmt.var: iter_var }
    #     # merge variables initialized before the block that
    #     # are updated in the body of the loop
    #     live_in, _ = stmt.attribs[LiveVarAnalysis.analysis_name]
    #     _, def_out = stmt.body.attribs[DefinitionAnalysis.analysis_name]
    #     # generate fresh variables for all changed variables
    #     changed_vars: set[str] = live_in & def_out
    #     changed_map: dict[str, str] = dict()
    #     for name in changed_vars:
    #         t = self.gensym.fresh(name)
    #         changed_map[name] = t
    #     # create the new context for the loop
    #     loop_ctx: None = dict()
    #     for name in ctx:
    #         if name in changed_map:
    #             loop_ctx[name] = changed_map[name]
    #         else:
    #             loop_ctx[name] = ctx[name]
    #     # compile the loop body using the loop context
    #     body, body_ctx = self._visit_block(stmt.body, loop_ctx: None)
    #     # merge all changed variables using phi nodes
    #     phis: list[ir.PhiNode] = []
    #     for name, t in changed_map.items():
    #         old_name = ctx[name]
    #         new_name = body_ctx[name]
    #         assert old_name != new_name, 'must be different by definition analysis'
    #         phis.append(ir.PhiNode(t, old_name, new_name, ir.AnyType()))
    #     # create new statement and context
    #     s = ir.ForStmt(iter_var, ir.AnyType(), cond, body, phis)
    #     new_ctx: None = dict()
    #     for name in ctx:
    #         if name in changed_map:
    #             new_ctx[name] = changed_map[name]
    #         else:
    #             new_ctx[name] = ctx[name]
    #     return s, new_ctx

    # def _visit_context(self, stmt, ctx: None):
    #     if stmt.name is not None:
    #         t = self.gensym.fresh(stmt.name)
    #         ctx = { **ctx, stmt.name: t }
    #         body, new_ctx = self._visit(stmt.body, ctx: None)
    #         return ir.ContextStmt(t, stmt.props, body), new_ctx
    #     else:
    #         body, new_ctx = self._visit(stmt.body, ctx: None)
    #         return ir.ContextStmt(None, stmt.props, body), new_ctx


class IRCodegen:
    """Lowers a FPy AST to FPy IR."""

    @staticmethod
    def lower(f: FunctionDef) -> ir.FunctionDef:
        return _IRCodegenInstance(f).lower()
