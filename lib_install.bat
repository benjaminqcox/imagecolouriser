@echo off


:start
cls

set python_ver=36

python ./get-pip.py

cd \
cd \python%python_ver%\Scripts\
pip install tensorflow
pip install opencv-python
pip install pillow

pause
exit