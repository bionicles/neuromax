from subprocess import check_output
from datetime import datetime
import pytz

# https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior
FORMAT = "%X %p %Z %A %-d %B %Y"
TIMEZONE = "US/Eastern"


def get_timestamp():
    stamp = pytz.timezone(TIMEZONE).localize(datetime.now()).strftime(FORMAT)
    hash = check_output('git describe --always', shell=True).decode().rstrip()
    return f'{stamp} on git {hash}'


# http://pytz.sourceforge.net/
TIMEZONE_LIST = pytz.common_timezones  # pytz.all_timezones


def show_timezones():
    for k, tz in enumerate(TIMEZONE_LIST):
        print(f"{k}: tz")
