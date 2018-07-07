#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 bzhou <bzhou@server2>
#
# Distributed under terms of the MIT license.
import time

class Logger:
    def __init__(self, logpath):
        self.f = open(logpath, 'w')
        self.data = []

    def log(self, info):
        info = time.strftime('%Y-%b-%d@%H:%M:%S--') + info
        print(info)
        self.f.write(info + '\n')
        self.f.flush()

