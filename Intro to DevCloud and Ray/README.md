## Ray Framework and Intel&reg; AI DevCloud

Install [Ray Framework](https://github.com/ray-project/ray):

```pip install --user Ray --ignore-installed funcsigs```

The Intel DevCloud account can be created on https://software.intel.com/en-us/devcloud/oneapi.
Connect to DevCloud Login node using the steps mentioned in https://devcloud.intel.com/datacenter/learn/connect-with-ssh-linux-macos/.
Obtain DevCloud compute node using:

```ssh devcloud```

```qsub -I -lselect=1```

```python ray_tutorial.py```

