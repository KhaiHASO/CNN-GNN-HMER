import os
import sys

# Add current folder to path to import tokenizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from latex_tokenizer import tokenize

# Predefined set of known LaTeX commands for mathematical expressions
KNOWN_COMMANDS = {
    # Structure & functions
    '\\frac', '\\sqrt', '\\sum', '\\int', '\\sin', '\\cos', '\\log', '\\lim', '\\cosh', '\\ln', '\\exp', '\\tan', '\\cot', '\\sec', '\\csc', '\\sinh', '\\tanh', '\\det',
    # Greek letters (lowercase)
    '\\alpha', '\\beta', '\\gamma', '\\delta', '\\epsilon', '\\zeta', '\\eta', '\\theta', '\\iota', '\\kappa', '\\lambda', '\\mu', '\\nu', '\\xi', '\\pi', '\\rho', '\\sigma', '\\tau', '\\upsilon', '\\phi', '\\chi', '\\psi', '\\omega', '\\varepsilon', '\\ell',
    # Greek letters (uppercase)
    '\\Alpha', '\\Beta', '\\Gamma', '\\Delta', '\\Epsilon', '\\Zeta', '\\Eta', '\\Theta', '\\Iota', '\\Kappa', '\\Lambda', '\\Mu', '\\Nu', '\\Xi', '\\Pi', '\\Rho', '\\Sigma', '\\Tau', '\\Upsilon', '\\Phi', '\\Chi', '\\Psi', '\\Omega',
    # Operators & relations
    '\\times', '\\cdot', '\\approx', '\\infty', '\\partial', '\\nabla', '\\pm', '\\mp', '\\equiv', '\\neq', '\\leq', '\\geq', '\\div', '\\sim', '\\star', '\\ast', '\\dagger', '\\hbar', '\\in', '\\notin', '\\subset', '\\subseteq', '\\cap', '\\cup', '\\forall', '\\exists',
    # Arrows
    '\\to', '\\rightarrow', '\\leftarrow', '\\leftrightarrow', '\\longleftrightarrow', '\\uparrow', '\\downarrow', '\\implies', '\\iff',
    # Accents & decorations
    '\\hat', '\\bar', '\\tilde', '\\check', '\\acute', '\\grave', '\\ddot', '\\dddot', '\\vec', '\\prime',
    # Fonts & formatting
    '\\mathrm', '\\mathbf', '\\mathcal', '\\cal', '\\mathfrak', '\\mathbb', '\\text', '\\operatorname', '\\operatorname*',
    # Spacing & ellipsis
    '\\quad', '\\qquad', '\\dots', '\\cdots', '\\vdots', '\\ddots',
    # Brackets & fences
    '\\left', '\\right', '\\left(', '\\right)', '\\left[', '\\right]', '\\left\\{', '\\right\\}', '\\right.', '\\left.', '\\left|', '\\right|', '\\Big', '\\big', '\\Bigg', '\\bigg', '\\langle', '\\rangle', '\\left\\Vert', '\\right\\Vert'
}

class ASTNode:
    """Node for Abstract Syntax Tree."""
    def __init__(self, type_, value=None, children=None):
        self.type = type_           # 'expression', 'frac', 'sqrt', 'sup', 'sub', 'group', 'token'
        self.value = value           # token dict if token
        self.children = children or []

class LatexParser:
    """Recursive descent parser to validate LaTeX and build an AST."""
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def peek(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self):
        tok = self.peek()
        if tok:
            self.pos += 1
        return tok

    def parse(self):
        nodes = []
        while self.pos < len(self.tokens):
            tok = self.peek()
            if tok['type'] in ('brace_close', 'paren_close', 'bracket_close'):
                break
            term = self.parse_term()
            if term:
                nodes.append(term)
        return ASTNode('expression', children=nodes)

    def parse_term(self):
        tok = self.peek()
        if not tok:
            return None
            
        # 1. Unknown command check
        if tok['type'] == 'command':
            cmd = tok['token']
            # strip trailing spaces or normalize command name
            if cmd not in KNOWN_COMMANDS and not cmd.startswith('\\left') and not cmd.startswith('\\right') and not cmd.startswith('\\begin') and not cmd.startswith('\\end'):
                raise SyntaxError(f"unknown_command: Unknown LaTeX command {cmd}")
        
        # 2. Open Braces: { expr }
        if tok['type'] == 'brace_open':
            self.consume() # consume '{'
            expr = self.parse()
            end_tok = self.peek()
            if not end_tok or end_tok['type'] != 'brace_close':
                raise SyntaxError("bracket_mismatch: Missing closing brace }")
            self.consume() # consume '}'
            if len(expr.children) == 0:
                raise SyntaxError("empty_group_error: Found empty braces {}")
            return ASTNode('group', value='{}', children=[expr])
            
        # 3. Open Parentheses: ( expr )
        elif tok['type'] == 'paren_open':
            self.consume()
            expr = self.parse()
            end_tok = self.peek()
            if not end_tok or end_tok['type'] != 'paren_close':
                raise SyntaxError("bracket_mismatch: Missing closing parenthesis )")
            self.consume()
            return ASTNode('group', value='()', children=[expr])

        # 4. Open Brackets: [ expr ]
        elif tok['type'] == 'bracket_open':
            self.consume()
            expr = self.parse()
            end_tok = self.peek()
            if not end_tok or end_tok['type'] != 'bracket_close':
                raise SyntaxError("bracket_mismatch: Missing closing bracket ]")
            self.consume()
            return ASTNode('group', value='[]', children=[expr])

        # 5. Fraction command: \frac { num } { den }
        elif tok['token'] == '\\frac':
            self.consume()
            # Numerator argument
            numerator = self.parse_argument()
            if not numerator:
                raise SyntaxError("frac_argument_error: \\frac is missing numerator")
            # Denominator argument
            denominator = self.parse_argument()
            if not denominator:
                raise SyntaxError("frac_argument_error: \\frac is missing denominator")
            return ASTNode('frac', children=[numerator, denominator])

        # 6. Square root command: \sqrt [ degree ] { body } or \sqrt { body }
        elif tok['token'] == '\\sqrt':
            self.consume()
            opt = None
            next_tok = self.peek()
            if next_tok and next_tok['type'] == 'bracket_open':
                opt = self.parse_argument()
            body = self.parse_argument()
            if not body:
                raise SyntaxError("sqrt_argument_error: \\sqrt is missing mandatory body argument")
            children = [body] if not opt else [opt, body]
            return ASTNode('sqrt', children=children)
            
        # 7. Superscript or Subscript: ^ or _
        elif tok['type'] in ('sup', 'sub'):
            op_tok = self.consume()
            arg = self.parse_argument()
            if not arg:
                raise SyntaxError(f"sup_sub_error: Operator {op_tok['token']} is missing its argument")
            type_ = 'sup' if op_tok['type'] == 'sup' else 'sub'
            return ASTNode(type_, children=[arg])
            
        else:
            # Simple token
            self.consume()
            return ASTNode('token', value=tok)

    def parse_argument(self):
        tok = self.peek()
        if not tok:
            return None
        if tok['type'] in ('brace_open', 'bracket_open'):
            return self.parse_term()
        elif tok['type'] in ('brace_close', 'paren_close', 'bracket_close'):
            return None
        elif tok['type'] in ('sup', 'sub'):
            return None
        else:
            self.consume()
            return ASTNode('token', value=tok)

def validate_latex(latex_str, render_success=1):
    """Validate LaTeX syntax rules and return status, error type, and message."""
    if render_success == 0:
        return {
            "is_valid": False,
            "error_type": "render_error",
            "error_message": "LaTeX render check failed by matplotlib.",
            "num_errors": 1
        }
        
    try:
        tokens = tokenize(latex_str)
    except Exception as e:
        return {
            "is_valid": False,
            "error_type": "tokenizer_error",
            "error_message": str(e),
            "num_errors": 1
        }
        
    parser = LatexParser(tokens)
    try:
        parser.parse()
        # Handle leftover unmatched closing brackets
        if parser.pos < len(tokens):
            tok = tokens[parser.pos]
            if tok['type'] in ('brace_close', 'paren_close', 'bracket_close'):
                raise SyntaxError(f"bracket_mismatch: Unmatched closing {tok['token']}")
            else:
                raise SyntaxError(f"parsing_error: Trailing unparsed tokens starting with {tok['token']}")
    except SyntaxError as e:
        msg = str(e)
        err_type = "parsing_error"
        for possible_type in ["bracket_mismatch", "frac_argument_error", "sqrt_argument_error", "sup_sub_error", "unknown_command", "empty_group_error"]:
            if msg.startswith(possible_type):
                err_type = possible_type
                msg = msg.split(":", 1)[1].strip() if ":" in msg else msg
                break
        return {
            "is_valid": False,
            "error_type": err_type,
            "error_message": msg,
            "num_errors": 1
        }
    except Exception as e:
        return {
            "is_valid": False,
            "error_type": "other_error",
            "error_message": str(e),
            "num_errors": 1
        }
        
    return {
        "is_valid": True,
        "error_type": "none",
        "error_message": "",
        "num_errors": 0
    }

if __name__ == "__main__":
    tests = [
        r"\frac{x+1}{y}",                    # Valid
        r"\frac{x+1}",                       # Missing denom
        r"x^{2} + y_{j}",                    # Valid
        r"x^",                               # Missing sup arg
        r"\sqrt{a+b}",                       # Valid
        r"\sqrt",                            # Missing body
        r"\frax{a}{b}",                      # Unknown command
        r"\frac{x+1}{}",                     # Empty group
        r"{a+b)"                             # Mismatched bracket
    ]
    
    print("Testing validator:")
    for t in tests:
        res = validate_latex(t)
        print(f"LaTeX: {t} -> Valid: {res['is_valid']}, Error: {res['error_type']} ({res['error_message']})")
