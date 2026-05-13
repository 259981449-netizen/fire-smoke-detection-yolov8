import os
import sys

pkgs = ['ultralytics', 'PyQt5', 'pyqt5-tools==5.15.2.3.1']

for each in pkgs:
    cmd_line = f'"{sys.executable}" -m pip install {each} -i https://pypi.tuna.tsinghua.edu.cn/simple'
    os.system(cmd_line)
