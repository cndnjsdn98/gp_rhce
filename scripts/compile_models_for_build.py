#!/usr/bin/env python

from src.quad_opt.quad_optimizer_mhe import init_compile as mhe_compile
from src.quad_opt.quad_optimizer_mpc import init_compile as mpc_compile

if __name__ == "__main__":
    mhe_compile()
    mpc_compile()