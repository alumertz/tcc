# To run

## Before running:
python3 -m venv mlenv
source mlenv/bin/activate

pip install pandas

## If you need to process data again: 
python3 src/process_data.py

## Binary classification (cancer vs passenger)
python3 src/main.py

## Multiclass classification (TSG vs Oncogene vs passenger)  
python3 src/main.py -multiclass

## Use Renan's original data format
python3 src/main.py -renan

# Combine both options
python3 src/main.py -multiclass -renan


# Data info
2class: Binary classification (1=cancer, NaN=candidate)
3class: Multiclass classification (1 = TSG, 2 = Oncogene, NaN=candidate)