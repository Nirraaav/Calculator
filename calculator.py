#!/usr/bin/env python3

import math
import operator
from typing import Any, List, Optional, Union
import sys

from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.styles import Style
from prompt_toolkit.shortcuts import print_formatted_text
from prompt_toolkit.formatted_text import PygmentsTokens
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.keys import Keys
from prompt_toolkit.key_binding import KeyBindings
from pygments.lexers.python import PythonLexer

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

# Define the autocomplete words
# autocomplete_words = list(set(math.__dict__.keys())) + list(set(operator.__dict__.keys()))
autocomplete_words = list(set(["sin", "cos", "tan", "sqrt", "ln", "log", "log2", "asin", "acos", "atan", "atan2", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "factorial", "fac", "phi", "gamma", "c", "G", "break", "end", "clear"]))
autocomplete_words = list(set(autocomplete_words))

# Define the style for the autocomplete suggestions
style = Style.from_dict({
    'prompt': '#00ff00 bold', 
    'autocomplete': '#ansigreen',
})

history = InMemoryHistory()

# Set up the WordCompleter for autocomplete
completer = WordCompleter(autocomplete_words, ignore_case=True)

# Function to get the input with autocompletion
def get_input_with_autocomplete() -> str:
    return prompt('>>> ', lexer=PygmentsLexer(PythonLexer), completer=completer, style=style, history=history)

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
    atom    :: PI | E | TAU | PHI | GAMMA | C | G | real | fn '(' expr ')' | '(' expr ')'
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
        tau = CaselessKeyword("TAU")
        phi = CaselessKeyword("PHI")
        gamma = CaselessKeyword("GAMMA")
        c = CaselessKeyword("C")
        g = CaselessKeyword("G")
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
        g_ = fn_call | pi | e | tau | phi | gamma | c | g | fnumber | ident  # type: ignore
        assert g_ is not None
        # atom = addop[...] + (
        #     g_.setParseAction(push_first) | Group(lpar + expr + rpar)  # type: ignore
        # )
        atom = addop[...] + (
            g_ | Group(lpar + expr + rpar) | g_.setParseAction(push_first)
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
    "fac": math.factorial,
}

constants = {
    "PI": math.pi,
    "E": math.e,
    "TAU": math.tau,
    "PHI": (1 + math.sqrt(5)) / 2,  # Golden ratio
    "GAMMA": 0.57721566490153286060,  # Euler-Mascheroni constant
    "C": 299792458,  # Speed of light in meters per second
    "G": 6.67430e-11,  # Gravitational constant in m^3 kg^-1 s^-2
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
    elif op in constants:
        return constants[op]
    elif op in fn:
        # note: args are pushed onto the stack in reverse order
        args = reversed([evaluate_stack(s) for _ in range(num_args)])
        return fn[op](*args)
    elif op == "factorial" or op == "fac":
        return math.factorial(int(evaluate_stack(s)))
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

# Add command history support
kb = KeyBindings()

@kb.add('up')
def _(event):
    # Move up in the command history
    event.current_buffer.auto_up()

@kb.add('down')
def _(event):
    # Move down in the command history
    event.current_buffer.auto_down()


try:
    while True:
        inp = get_input_with_autocomplete()
        if inp.lower() == "end" or inp.lower() == "break" or inp.lower() == "clear":
            break
        result = expr_compute(inp)

        if result is not None:
            print(f"Result = {result}")
        else:
            print("Invalid input or empty expression.")

except (KeyboardInterrupt, EOFError):
    print("\nExiting...")
    sys.exit(0)

# flake8: noqa
