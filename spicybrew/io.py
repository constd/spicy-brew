__author__ = 'const'
from json import load, dump


def jread(filename):
    with open(filename, 'rb') as f:
        data = load(f)
    return data


def jwrite(data, filename, write_mode='wb'):
    with open(filename, write_mode) as f:
        dump(data, f, indent=4, separators=(',', ': '))

