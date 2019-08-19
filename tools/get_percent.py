from .get_value import get_value

DIGITS = 1


def get_percent(total, subtotal, digits=DIGITS):
    percentage = round((get_value(subtotal) / get_value(total)) * 100., digits)
    return f"{percentage}%"
