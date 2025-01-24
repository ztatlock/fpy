"""Comparison operators"""

from enum import Enum

class CompareOp(Enum):
    """Comparison operators as an enumeration"""
    LT = 0
    LE = 1
    GE = 2
    GT = 3
    EQ = 4
    NE = 5

    def symbol(self):
        """Get the symbol for the operator"""
        return _symbol_table[self]

_symbol_table = {
    CompareOp.LT: '<',
    CompareOp.LE: '<=',
    CompareOp.GE: '>=',
    CompareOp.GT: '>',
    CompareOp.EQ: '==',
    CompareOp.NE: '!='
}
