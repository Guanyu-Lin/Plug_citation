import argparse
import os
import torch
import logging

from tools.init_tool import init_all
from config_parser import create_config
from tools.train_tool import train
import bmtrain as bmt
from bmtrain import print_rank
import re