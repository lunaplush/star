import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys


class Person:
    """
    HELLO
    """
    def __init__(self, name):
        self.greeting = "<div>Hello {name}</div>"
        self.name = name
    def __str__(self):
        return self.make_greeting()

    def make_greeting(self):
        return self.greeting.format(name=self.name)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('integers', metavar='N', type=int, nargs='+',
                        help='an integer for the accumulator')
    parser.add_argument('--sum', dest='accumulate', action='store_const',
                        const=sum, default=max,
                        help='sum the integers (default: find the max)')

    args = parser.parse_args()
    print(args.accumulate(args.integers))
    people = [
        Person("A"),
        Person("B")
    ]
    print(sys.argv)