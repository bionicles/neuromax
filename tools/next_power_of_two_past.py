def next_power_of_two_past(n):
    p = 1
    while p < n:
        p = p * 2
    return p
