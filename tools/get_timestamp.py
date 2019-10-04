from datetime import datetime
import pytz

# https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior
FORMAT = "%d %B %Y (%A) by BAH on HHI 16.04 TF 2.0 GTX 1070 at %X %p %Z"
TIMEZONE = "US/Eastern"


def get_timestamp():
    return pytz.timezone(TIMEZONE).localize(datetime.now()).strftime(FORMAT)


# http://pytz.sourceforge.net/
TIMEZONE_LIST = pytz.common_timezones  # pytz.all_timezones


def show_timezones():
    for k, tz in enumerate(TIMEZONE_LIST):
        print(f"{k}: tz")
