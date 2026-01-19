
## Using a multi-column tabular stroke dataset from Kaggle to train a KAN network and maybe a LiquidNN network for classification.

Access at: ``LINK`` to see my analysis

## INSTRUCTIONS FOR DEVS:

## Install python (check https://www.python.org/downloads/)

## Install uv  
```powershell
## WINDOWS
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

## LINUX
curl -LsSf https://astral.sh/uv/install.sh | sh

Install dependencies (uv creates a separate venv in cwd):

uv sync
```
## Run the program:
```python
python -u main.py
```