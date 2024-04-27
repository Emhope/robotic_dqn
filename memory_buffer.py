#! /usr/bin/env python3
import sys
import random


class MemoryBuffer:
    def __init__(self, size):
        self.data = []
        self.size = size
        self.full = False
    
    def push(self, item):
        if len(self.data) == self.size:
            self.data.pop(0)
            self.full = True
        self.data.append(item)
    
    def clear(self):
        self.data = []
        self.full = False
    
    def sample(self, size):
        return random.sample(self.data, size)

    def __sizeof__(self):
        s = 0
        for d in self.data:
            if isinstance(d, list):
                for i in d:
                    s += sys.getsizeof(i)
            s += sys.getsizeof(d)
        return s
    