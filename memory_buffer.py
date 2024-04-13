#! /usr/bin/env python3

import random


class MemoryBuffer:
    def __init__(self, size):
        self.data = []
        self.size = size
        self.updates = 0
    
    def push(self, item):
        self.updates += 1
        if len(self.data) == self.size:
            self.data.pop(0)
        self.data.append(item)
    
    def clear(self):
        self.data = []
    
    def sample(self):
        self.updates = 0
        return random.sample(self.data, len(self.data))
    
    def is_fresh(self):
        return self.updates > self.size
    