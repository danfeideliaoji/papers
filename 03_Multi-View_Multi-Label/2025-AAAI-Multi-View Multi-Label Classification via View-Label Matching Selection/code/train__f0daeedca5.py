"""兼容入口，与 demo.py 使用完全相同的修正版训练流程。"""
import runpy

if __name__ == '__main__':
    runpy.run_module('demo', run_name='__main__')
