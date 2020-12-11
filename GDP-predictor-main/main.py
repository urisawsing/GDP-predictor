#!/usr/bin/env python
import os
import logging
import argparse
from datetime import datetime
import analysis as an

from utils import config, io, models

numind=utils.config.NUM_PREDICTORS
countries=an.countrylist.countryarray()
for country in countries[0]:
    name=country
    indicators=an.select.selection(name,numind)
    