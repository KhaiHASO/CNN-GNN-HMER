import re

# Specifications for math LaTeX tokens
TOKEN_SPECIFICATION = [
    ('COMMAND', r'\\[a-zA-Z]+\*?'),                 # LaTeX commands like \frac, \sqrt, \alpha, \operatorname*
    ('NUMBER', r'\d+(?:\.\d+)?'),                   # Integer or decimal number
    ('BRACE_OPEN', r'\{'),                           # Left brace
    ('BRACE_CLOSE', r'\}'),                          # Right brace
    ('PAREN_OPEN', r'\('),                          # Left parenthesis
    ('PAREN_CLOSE', r'\)'),                         # Right parenthesis
    ('BRACKET_OPEN', r'\['),                        # Left bracket
    ('BRACKET_CLOSE', r'\]'),                       # Right bracket
    ('SUP', r'\^'),                                 # Superscript
    ('SUB', r'_'),                                  # Subscript
    ('OPERATOR', r'[+\-*=\/<>!,.]'),                # Operators and punctuation
    ('VARIABLE', r'[a-zA-Z]'),                      # Variables
    ('WHITESPACE', r'\s+'),                          # Skip spaces
    ('MISC', r'.'),                                 # Any other character
]

def tokenize(latex_str):
    """Tokenize a math LaTeX string into a list of dictionaries with type, token, and position."""
    tok_regex = '|'.join(f'(?P<{name}>{pattern})' for name, pattern in TOKEN_SPECIFICATION)
    tokens = []
    
    for mo in re.finditer(tok_regex, latex_str):
        kind = mo.lastgroup
        value = mo.group()
        if kind == 'WHITESPACE':
            continue
        tokens.append({
            "token": value,
            "type": kind.lower(),
            "position": mo.start()
        })
    return tokens

if __name__ == "__main__":
    test_str = r"\frac{x^{2} + 15}{4} = \sqrt{y}"
    print(f"Testing tokenizer on: {test_str}")
    toks = tokenize(test_str)
    for t in toks[:10]:
        print(t)
