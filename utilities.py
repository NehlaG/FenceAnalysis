import numpy as np
import os

def sortKey(fname):
    fname = os.path.splitext(fname)[0]
    # print('fname: ', fname)
    # split_fname = fname.split('_')
    # print('split_fname: ', split_fname)
    nums = [int(s) for s in fname.split('_') if s.isdigit()]
    try:
        key = nums[0]
    except IndexError:
        return fname
    # print('key: ', key)
    return key

def processArguments(args, params):
    # arguments specified as 'arg_name=argv_val'
    no_of_args = len(args)
    for arg_id in range(no_of_args):
        arg = args[arg_id].split('=')
        if len(arg) != 2 or not arg[0] in params.keys():
            print('Invalid argument provided: {:s}'.format(args[arg_id]))
            return
        params[arg[0]] = type(params[arg[0]])(arg[1])
