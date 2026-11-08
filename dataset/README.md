# Dataset Commands

Run from the bundle root with Python 3.11:

~~~powershell
py -3.11 -m venv .\dataset\.venv
.\dataset\.venv\Scripts\python.exe -m pip install -r .\dataset\requirements.txt
.\run.ps1 build --archive 'D:\XTraffic-v8' --output '.\Data_rebuilt'
~~~

The archive must contain `node_order.npy` and `year_2023\year_2023\2023_p01.npy` through `2023_p12.npy`. Use a new output directory.

To use an existing 120-minute population:

~~~powershell
.\run.ps1 build --population 'D:\full_year_dataset\common\120min' --output '.\Data_rebuilt'
~~~

To generate the upstream population as well:

~~~powershell
.\dataset\.venv\Scripts\python.exe -m pip install -r .\dataset\requirements-upstream.txt
.\run.ps1 build --archive 'D:\XTraffic-v8' --rebuild-population 'D:\CA-TID-population' --output '.\Data_rebuilt'
~~~

This last command also needs `sensor_meta_feature.csv`, `incidents_y2022.csv`, and `incidents_y2023.csv` in the archive.

Run experiments on the new data with `--data-root '.\Data_rebuilt'`.
