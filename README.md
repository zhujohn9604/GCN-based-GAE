# GCN-based-GAE

The repository implements GCN-based GAE for monitoring technology convergence. The **example.py** demonstrates how to use GCN to generate patent and technological keywords vectors and technology intensity coordinates.


### With pip

This repository is built on the Package gdl version 0.4.3.

You should install 🤗 Transformers in a [virtual environment](https://docs.python.org/3/library/venv.html). If you're unfamiliar with Python virtual environments, check out the [user guide](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).

First, create a virtual environment with the version of Python you're going to use and activate it.

Then, you will need to install at least one of Flax, PyTorch or TensorFlow.
Please refer to [TensorFlow installation page](https://www.tensorflow.org/install/), [PyTorch installation page](https://pytorch.org/get-started/locally/#start-locally) and/or [Flax installation page](https://github.com/google/flax#quick-install) regarding the specific install command for your platform.

When one of those backends has been installed, 🤗 Transformers can be installed using pip as follows:

```bash
pip install transformers
```
