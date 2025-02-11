from random import randint, choice, sample, shuffle, random

def get_reverse_add(a, b):
    s = str(int(a) + int(b))[::-1]
    l = max(len(a), len(b))
    s = s.ljust(l+1, '0')
    return f'{a[::-1]}+{b[::-1]}=', s, None

def get_reverse(a):
    return a + '=', a[::-1], None

def get_copy(a):
    return f'{a}=', a, None

def get_forward(a, b):
    s = str(int(a) + int(b))
    l = max(len(a), len(b)) # match the setting with reverse
    s = s.rjust(l+1, '0')

    return f'{a}+{b}=', s, None

def get_forward2(a, b):
    s = str(int(a) + int(b))

    return f'{a}+{b}=', s, None

def split_digits_mult(a, b):
    a_digits = []
    b_digits = []

    while True:
        if a > 0:
            a_digits.append(a % 10)
        if b > 0:
            b_digits.append(b % 10)
        # a_digits.append(a % 10)
        # b_digits.append(b % 10)
        a //= 10
        b //= 10
        if a == 0 and b == 0:
            break
    return a_digits, b_digits


def get_COT_mult(a, b):
    a, b = int(a), int(b)
    a_digits, b_digits = split_digits_mult(a, b)

    a_str = ''.join(map(str, a_digits))
    b_str = ''.join(map(str, b_digits))

    len_a, len_b = len(a_str), len(b_str)

    max_len = max(len(a_str), len(b_str))

    prompt = ''.join(map(str, a_digits)) + '*' + ''.join(map(str, b_digits)) + '='
    cot = ''
    ans = str(a * b)[::-1]
    result_up_to_i = 0

    for i, b_i in enumerate(b_digits):
        c_i = b_i * a * 10**i

        result_up_to_i += c_i 

        c_i_rev = str(c_i)[::-1]
        c_i_rev = c_i_rev.ljust(i + len_a + 1, '0')

        if i == 0:
            cot += c_i_rev
        else:
            cot += '+' + c_i_rev
            if i < len_b - 1:
                cot += '(' + str(result_up_to_i)[::-1].ljust(i + len_a + 1, '0') + ')'
    
    cot += '=' + ans.ljust(len_a+len_b, '0')


    return prompt, cot, None
