# `tipeval`

`tipeval` is a package that can be used for evaluating 3D images of nanoindentation tips to determine
the area function and tip radius. The 3D images can in principle be recorded with any measurement tool 
including 3D laser confocal scanning microscopy, atomic force microscopy or scanning probe microscopy 
using the nanoindentation tip itself (called *self imaging*). It can be used from the command line but for 
choosing the data for fitting a graphical user interface is used. There is also the plan to build a 
graphical user interface for the whole evaluation but this is not currently fully working.

## Installation
At first, clone the repository from gitlab https://git.unileoben.ac.at/. Search for and 
go to the `tipeval` project and clone it onto your computer. If you have never cloned a git
repository before you will find plenty of introductions online. Just one thing: to install git in your 
conda environment use

```conda install -c anaconda git```

in the environment you later use for installing `tipeval` (see next step). 

In order to be able to use `tipeval`, it is best to at first have a working anaconda distribution 
on the computer. Anaconda can be downloaded [here](https://www.anaconda.com/). Since 
`tipeval` depends on many third party packages, the best thing would be to make a dedicated
`conda` environment for it. I recommend to use python 3.9. This can be done by typing 
the following line into the command line:

```conda create -n <environment_name> python=3.9``` 

where `<environment_name>` should be replaced by the desired name. 
Alternatively to using anaconda you can of course also use
a different python virtual environment tool such as `venv`. After generating
the environment activate it with:

```conda activate <environment_name>```.

Now, go into the folder where you have downloaded `tipeval` and pip-install the requirements using:

```pip install -r requirements.txt```.

This will install all required third party packages with the correct version. If you 
do not have pip available, download and install it from the 
[pip installation homepage](https://pip.pypa.io/en/stable/installation/). 

Finally you need to install `tipeval`. For this use the command line to move to the directory 
where the `setup.py` file is and run 

```pip install .```

(the dot stands for *current directory*). This will install `tipeval` as a package in the current 
environment. If you want to modify the installation later on (for instance to improve the code or
fix errors) you need to install it as 'editable'. For that you need to add the `-e` flag during 
installation:

```pip install -e .```

### Packages that can be difficult to install
There are two packages that might be difficult to install. One is `mayavi` and the other 
is `vtk` which `mayavi` depends on. Using the `requirements.txt` file there should not be a 
problem but if one occurs with either of the two mentioned packages then try to 
install them separately using `pip`, first installing `vtk`. As of now (March 2022) 
`vtk` does not work with Python 3.10 which is why I recommend Python 3.9. 

## Usage
The usage of `tipeval` is demonstrated in two accompanying `jupyter` notebooks. Example data sets of 
tip images recorded with self-imaging and laser confocal scanning microscopy are saved in the examples 
folder. Documentation of the code is currently unfortunately not complete. In order to be able to run 
the example notebook you will need to install jupyter in the same `conda` environment you installed 
`tipeval` using `pip install jupyter`. After installation go to the docs-folder and look at the two 
notebooks, starting with the basic one first.