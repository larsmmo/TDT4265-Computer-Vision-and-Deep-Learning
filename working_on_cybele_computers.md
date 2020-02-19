# Working on Cybele Computers

There are 25 computers in Cybele, all with powerful GPU cards (1080, 1080ti etc). There are a lot of students in the class and you have to consider your fellow students when you are using the GPU resources.

We ask you to follow these rules when using the computers:

1. Each group can only use a single computer at a time.
2. It is not allowed to remote access the computers in school time. This is the time 08:00-20:00 every weekday (Monday-Friday).
3. Before you start utilizing the GPU, check that no one is using it with the command `nvidia-smi`.


## Environment
Every computer in the cybele lab comes with python2 and python3.
You can run code in python3 by using, `python3 my_program.py`.

### Installing packages
If you want to install additional packages in your environment, install it locally. For example:

```bash
pip3 --user install package_name
```

To get a list of already installed packages, use:
```
pip list
```


## Working remote
You can access the computer from your home computer/laptop by using ssh. To connect, you can connect to the following adress:

```
ssh [ntnu-username]@clab[00-25].idi.ntnu.no
```
For example, 
```
ssh haakohu@clab21.idi.ntnu.no
```

**NOTE**: It is not allowed to ssh onto the computers in school time! See above for rules
