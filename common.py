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

    def log(self, info):
        print(info)
        f.write(info + '\n')
        f.flush()

