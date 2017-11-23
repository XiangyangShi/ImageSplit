# ImageSplit

The python file: Nov20.py
is the latest version of my source code.

It can simpily run with command line.
But operating access is necessary.
So is Librares of { Numpy and OpenCV }

OpenCV3 is difficult to install.
Anaconda# conda install openCV3
is recommended

# environment
> install anaconda first
> check whether opencv3 is installed
```
conda list opencv
```
> ifnot install it
```
conda install -c https://conda.binstar.org/menpo opencv3
```

# How to operate:
```
git clone https://github.com/XiangyangShi/ImageSplit.git
cd ImageSplit
python Nov20.py -h
// for help
python Nov20.py -i 1
// for basic example
python Nov20.py -i 1 -f "jpgfilefullpath"
// for test single file
python Nov20.py -i 0
// test all file in the document of ./ImageSplit/Test
```
