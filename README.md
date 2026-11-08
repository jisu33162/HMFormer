# HMFormer

Open PowerShell in this folder and install the pinned experiment dependencies from `experiments/requirements.txt` using Python 3.12:

~~~powershell
py -3.12 -m venv .\experiments\.venv
.\experiments\.venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.\experiments\.venv\Scripts\python.exe -m pip install -r .\experiments\requirements.txt
~~~

Edit `parameter.txt` to set the experiment, data paths, seeds, model, training, and HMM/GMM parameters. Set `experiment` to `comparison`, `sensitivity`, or `ablation`, then run:

~~~powershell
.\run.ps1
~~~

Prepared datasets are in `Data`. Results are written to `outputs` unless `output` is set in the text file. Seeds default to `1 2 3 4 5`.

To use a different parameter file:

~~~powershell
.\run.ps1 --parameter-file '.\my_parameter.txt'
~~~

The Python equivalent is `python run.py`. See [dataset commands](dataset/README.md) to regenerate data.
