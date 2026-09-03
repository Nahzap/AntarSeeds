"""
Unicode-to-ASCII LUT (Look-Up Table)

Tabla centralizada para convertir caracteres Unicode científicos a ASCII seguro
en sistemas Windows con encoding cp1252.

Principio: ZERO hardcoding de strings. Toda conversión por tabla LUT.
"""

# ============================================================================
# UNICODE-TO-ASCII MAPPING TABLE
# ============================================================================
# Formato: {unicode_char: ascii_replacement}

UNICODE_LUT = {
    # Greek letters (lowercase)
    'λ': 'lambda',
    'α': 'alpha',
    'β': 'beta',
    'γ': 'gamma',
    'δ': 'delta',
    'ε': 'epsilon',
    'θ': 'theta',
    'μ': 'mu',
    'σ': 'sigma',
    'τ': 'tau',
    'φ': 'phi',
    'ψ': 'psi',
    'ω': 'omega',
    
    # Greek letters (uppercase)
    'Δ': 'Delta',
    'Σ': 'Sigma',
    'Ω': 'Omega',
    
    # Subscripts (common in scientific notation)
    '₀': '_0',
    '₁': '_1',
    '₂': '_2',
    '₃': '_3',
    '₄': '_4',
    '₅': '_5',
    '₆': '_6',
    '₇': '_7',
    '₈': '_8',
    '₉': '_9',
    
    # Superscripts
    '⁰': '^0',
    '¹': '^1',
    '²': '^2',
    '³': '^3',
    '⁴': '^4',
    '⁵': '^5',
    '⁶': '^6',
    '⁷': '^7',
    '⁸': '^8',
    '⁹': '^9',
    
    # Dashes and arrows
    '—': '-',
    '–': '-',
    '→': '->',
    '←': '<-',
    '↑': '^',
    '↓': 'v',
    '↔': '<->',
    '⇒': '=>',
    '⇐': '<=',
    
    # Math symbols
    '≈': '~=',
    '≠': '!=',
    '≤': '<=',
    '≥': '>=',
    '×': 'x',
    '÷': '/',
    '±': '+/-',
    '∞': 'inf',
    '∑': 'SUM',
    '∏': 'PROD',
    '∫': 'INT',
    '∂': 'd',
    '∇': 'nabla',
    '√': 'sqrt',
    
    # Checkmarks and symbols
    '✓': 'OK',
    '✗': 'X',
    '✔': 'OK',
    '✘': 'X',
    '⚠': '!',
    '⚡': '**',
    '★': '*',
    '☆': '*',
    '•': '-',
    '◦': 'o',
    '▪': '-',
    '▫': 'o',
    
    # Degree and special
    '°': 'deg',
    '℃': 'C',
    '℉': 'F',
    'Å': 'A',
    
    # Math sets
    'ℝ': 'R',
    'ℂ': 'C',
    'ℕ': 'N',
    'ℤ': 'Z',
    'ℚ': 'Q',
    
    # Misc scientific
    '∈': 'in',
    '∉': 'not_in',
    '∩': 'AND',
    '∪': 'OR',
    '⊂': 'subset',
    '⊃': 'superset',
    '∅': 'empty',
}


def sanitize_for_ascii(text: str, lut: dict = None) -> str:
    """
    Convierte texto Unicode a ASCII seguro usando tabla LUT.
    
    Args:
        text: String con posibles caracteres Unicode
        lut: Tabla LUT personalizada (opcional, usa UNICODE_LUT por defecto)
    
    Returns:
        String con solo caracteres ASCII seguros
    
    Examples:
        >>> sanitize_for_ascii("λ₁=0.221, Status=✓ HEALTHY")
        "lambda_1=0.221, Status=OK HEALTHY"
        >>> sanitize_for_ascii("Recall@1: 97.8% → 98.1%")
        "Recall@1: 97.8% -> 98.1%"
    """
    if lut is None:
        lut = UNICODE_LUT
    
    result = text
    for unicode_char, ascii_replacement in lut.items():
        result = result.replace(unicode_char, ascii_replacement)
    
    return result


def make_logger_ascii_safe(logger):
    """
    Envuelve los handlers de un logger para sanitizar Unicode automáticamente.
    
    Args:
        logger: logging.Logger instance
    
    Returns:
        Logger modificado con handlers ASCII-safe
    """
    import logging
    
    class ASCIISafeFormatter(logging.Formatter):
        """Formatter que sanitiza Unicode antes de formatear."""
        
        def __init__(self, fmt=None, datefmt=None, lut=None):
            super().__init__(fmt, datefmt)
            self.lut = lut or UNICODE_LUT
        
        def format(self, record):
            # Sanitizar mensaje
            if isinstance(record.msg, str):
                record.msg = sanitize_for_ascii(record.msg, self.lut)
            
            # Sanitizar args si existen
            if record.args:
                if isinstance(record.args, dict):
                    record.args = {
                        k: sanitize_for_ascii(str(v), self.lut) if isinstance(v, str) else v
                        for k, v in record.args.items()
                    }
                elif isinstance(record.args, tuple):
                    record.args = tuple(
                        sanitize_for_ascii(str(arg), self.lut) if isinstance(arg, str) else arg
                        for arg in record.args
                    )
            
            return super().format(record)
    
    # Aplicar a todos los handlers
    for handler in logger.handlers:
        old_formatter = handler.formatter
        if old_formatter:
            new_formatter = ASCIISafeFormatter(
                fmt=old_formatter._fmt if hasattr(old_formatter, '_fmt') else None,
                datefmt=old_formatter.datefmt
            )
        else:
            new_formatter = ASCIISafeFormatter()
        
        handler.setFormatter(new_formatter)
    
    return logger


def safe_file_write(file_handle, text: str, lut: dict = None):
    """
    Escribe texto a archivo con sanitización Unicode automática.
    
    Args:
        file_handle: File object abierto en modo write
        text: String a escribir
        lut: Tabla LUT (opcional)
    """
    sanitized = sanitize_for_ascii(text, lut)
    file_handle.write(sanitized)
