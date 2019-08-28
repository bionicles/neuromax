from .get_value import get_value
from .log import log

DIGITS = 1


def get_percent(total, subtotal, digits=DIGITS):
    subtotal = get_value(subtotal)
    total = get_value(total)
    try:
        percentage = round((subtotal / total) * 100., digits)
    except Exception as e:
        log('get_percent Exception:', e, color='red')
        return "NAN%"
    return f"{percentage}%"
