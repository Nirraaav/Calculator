import math
import operator
from typing import Any, List, Optional, Union

from pyparsing import (
    CaselessKeyword,
    Forward,
    Group,
    Literal,
    Regex,
    Suppress,
    Token,
    Word,
    alphanums,
    alphas,
    delimitedList,
)  # NOQA

exprStack: Any = []


def push_first(toks: List[Token]):
    exprStack.append(toks[0])


def push_unary_minus(toks: List[Token]):
    for t in toks:
        if t == "-":
            exprStack.append("unary -")
        else:
            break


bnf = None


def BNF() -> Any:
    """
    expop   :: '^'
    multop  :: '*' | '/'
    addop   :: '+' | '-'
    integer :: ['+' | '-'] '0'..'9'+
    atom    :: PI | E | real | fn '(' expr ')' | '(' expr ')'
    factor  :: atom [ expop factor ]*
    term    :: factor [ multop factor ]*
    expr    :: term [ addop term ]*
    """
    global bnf
    if not bnf:
        # use CaselessKeyword for e and pi, to avoid accidentally matching
        # functions that start with 'e' or 'pi' (such as 'exp'); Keyword
        # and CaselessKeyword only match whole words
        e = CaselessKeyword("E")
        pi = CaselessKeyword("PI")
        # tau = CaselessKeyword("TAU")
        fnumber = Regex(r"[+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?")
        ident = Word(alphas, alphanums + "_$")

        plus, minus, mult, div = map(Literal, "+-*/")
        lpar, rpar = map(Suppress, "()")
        addop = plus | minus
        multop = mult | div
        expop = Literal("^")
        factorial = Literal("!")

        expr = Forward()
        expr_list = delimitedList(Group(expr))

        # add parse action that replaces the function identifier with a (name, number of args) tuple
        def insert_fn_argcount_tuple(t: List[Any]):
            fn = t.pop(0)
            num_args = len(t[0])
            t.insert(0, (fn, num_args))

        f = ident + lpar - Group(expr_list) + rpar  # type: ignore
        fn_call = f.setParseAction(insert_fn_argcount_tuple)  # type: ignore
        g = fn_call | pi | e | fnumber | ident  # type: ignore
        assert g is not None
        atom = addop[...] + (
            g.setParseAction(push_first) | Group(lpar + expr + rpar)  # type: ignore
        )
        atom = atom.setParseAction(push_unary_minus)  # type: ignore

        # by defining exponentiation as "atom [ ^ factor ]..." instead of "atom [ ^ atom ]...", we get right-to-left
        # exponents, instead of left-to-right that is, 2^3^2 = 2^(3^2), not (2^3)^2.
        factor = Forward()
        factor <<= atom + (expop + factor).setParseAction(push_first)[...]  # type: ignore
        term = (
            factor + (multop + factor).setParseAction(push_first)[...]  # type: ignore
        )  # type: ignore
        expr <<= (
            term + (addop + term).setParseAction(push_first)[...]  # type: ignore
        )  # type: ignore
        bnf = expr
    return bnf


# map operator symbols to corresponding arithmetic operations
epsilon = 1e-12
opn = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "^": operator.pow,
}

fn = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "sqrt": math.sqrt,
    # "cbrt": math.cbrt,
    "ln": math.log,
    "log": math.log10,
    "log2": math.log2,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "asinh": math.asinh,
    "acosh": math.acosh,
    "atanh": math.atanh,
    "factorial": math.factorial,
}


def evaluate_stack(s: List[Any]) -> Union[int, float]:
    op, num_args = s.pop(), 0
    if isinstance(op, tuple):
        op, num_args = op
    if op == "unary -":
        return -evaluate_stack(s)
    if op in "+-*/^":
        # note: operands are pushed onto the stack in reverse order
        op2 = evaluate_stack(s)
        op1 = evaluate_stack(s)
        return opn[op](op1, op2)
    elif op == "PI":
        return math.pi  # 3.1415926535
    elif op == "E":
        return math.e  # 2.718281828
    # elif op == "TAU":
    #     return math.tau
    elif op in fn:
        # note: args are pushed onto the stack in reverse order
        args = reversed([evaluate_stack(s) for _ in range(num_args)])
        return fn[op](*args)
    elif op == "factorial":
        return math.factorial(evaluate_stack(s))
    elif op[0].isalpha():
        raise Exception(f"invalid identifier {op}")
    else:
        # try to evaluate as int first, then as float if int fails
        try:
            return int(op)
        except ValueError:
            return float(op)


def expr_compute(s: str) -> Optional[float]:
    if s == "":
        return None
    exprStack[:] = []
    BNF().parseString(s, parseAll=True)
    val = evaluate_stack(exprStack[:])
    return val


while True:
    inp = str(input())
    if inp.lower() == "end" or inp.lower() == "break":
        break
    print(f"Result = {expr_compute(inp)}")

# flake8: noqa