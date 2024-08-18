from datetime import datetime


def get_current_datetime():
    """
    Get the current date, time, and day of the week.

    This function returns the current date, time, and the corresponding weekday in the system's
    local time zone. The date and time are formatted according to the ISO 8601 standard, and the
    day of the week is included as a separate string.

    Returns:
        str: A string representation of the current date and time in the format 'YYYY-MM-DDTHH:MM:SS (Weekday)'

    Example:
        >>> get_current_datetime()
        '2024-08-17T12:34:56 (Saturday)'
    """
    now = datetime.now()
    return f"{now.isoformat()} ({now.strftime('%A')})"
