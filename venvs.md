# How to Recreate the Virtual Environment

Inspired from [this tutorial](https://scitas-data.epfl.ch/confluence/display/DOC/Python+Virtual+Environments), here is a short tutorial as to how one can use AtLoc on the SCITAS IZAR cluster.

## Choosing `python3`

At the beginning, one has to choose the Python version. I have tried both, and it appeared that Python 3 worked better since it was compatible with more modules. Moreover, there was no notable difference in the code executionm, so I guess it should be good.

```bash
[user@ucluster:~]$ module spider python
-------------------------------------------------------------------------
  python:
-------------------------------------------------------------------------
     Versions:
        python/2.7.16
        python/3.7.3
 
 
[user@cluster:~]$ module spider python/3.7.3
-------------------------------------------------------------------------
  python: python/3.7.3
-------------------------------------------------------------------------
 
    You will need to load all module(s) on any one of the lines below before the "python/3.7.3" module is available to load.
 
      gcc/7.4.0
      gcc/8.3.0
      intel/18.0.5
 
[user@cluster:~]$ module load gcc/7.4.0 python/3.7.3
[user@cluster:~]$ python --version
Python 3.7.3
```

## Creating the virtualenv

I chose to name the venv `atloc` to reflect which modules are imported.

```bash
[user@cluster:~]$ virtualenv --system-site-packages venvs/atloc
Using base prefix '/ssoft/spack/humagne/v1/opt/spack/linux-rhel7-x86_E5v4_Mellanox/gcc-7.4.0/python-3.7.3-5lm3vikrg4nq4tjhx76dgqy7zbt4kfam'
New python executable in /home/user/venvs/atloc/bin/python3.7
Also creating executable in /home/user/venvs/atloc/bin/python
Installing setuptools, pip, wheel...
done.
```

## Activating the virtualenv

To install packages, one first needs to activate the virtual environment.

```bash
[user@cluster:~]$ source venvs/atloc/bin/activate
(atloc) [user@cluster:~]$ module list
Currently Loaded Modules:
  1) gcc/7.4.0   2) python/3.7.3
```

## Installing the relevant packages

The file [environment.yml](AtLoc-master/environment.yml) contains a list of Conda dependencies for AtLoc. Since IZAR does not support Conda, I only kept the list of pip dependencies in the lower part of the file.

Importing happens as follows:

```bash
(atloc) [user@cluster:~]$ pip install --no-cache-dir backports-abc
```

It is important to use the `--no-cache-dir` option to ensure an existing pre-compiled copy of the python package is not used.

This was just an example. Here is the complete list of dependencies:

```yml
dependencies:
  - pip:
    - backports-abc==0.5
    - backports.functools-lru-cache==1.5
    - chardet==3.0.4
    - colour-demosaicing==0.1.4
    - colour-science==0.3.13
    - configparser==4.0.2
    - cycler==0.10.0
    - futures==3.3.0
    - idna==2.8
    - jsonpatch==1.24
    - jsonpointer==2.0
    - kiwisolver==1.1.0
    - matplotlib==2.2.4
    - numpy==1.16.5
    - opencv-python==4.1.2.30
    - pillow==6.1.0
    - protobuf==3.11.1
    - pyparsing==2.4.2
    - python-dateutil==2.8.0
    - pytz==2019.2
    - pyzmq==18.1.0
    - requests==2.22.0
    - scipy==1.2.2
    - singledispatch==3.4.0.3
    - six==1.12.0
    - subprocess32==3.5.4
    - tensorboardx==1.6
    - torch==0.4.1
    - torchfile==0.1.0
    - torchvision==0.2.0
    - tornado==5.1.1
    - transforms3d==0.3.1
    - urllib3==1.25.4
    - visdom==0.1.8.9
    - websocket-client==0.56.0
```

Note that I did not use the version number when installing `backports-abc` in the example above. I think it is not crucial to have the exact same versions, except in two cases:
* for `numpy`, I ran `pip install --no-cache-dir numpy==1.16.5`
* for `torch`, I had some issues at first with `numpy` compatibility, so I ran `pip install --no-cache-dir torch==0.4.1.post2`

I think that is all, I hope this is clear...
