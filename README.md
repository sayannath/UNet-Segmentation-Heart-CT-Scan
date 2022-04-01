# UNET Segmentation on CT Scan Images

### How to Train the Model

```
1. python3 -m pip install --user virtualenv (In Mac or Linux)
   python -m pip install --user virtualenv (In Windows) 
   
   [Make sure Python3 is installed in your system]

2. python3 -m venv env (In Mac or Linux)
   python -m venv env (In Windows) 
   
3. source env/bin/activate  (In Mac or Linux)
   .\env\Scripts\activate (In Windows) 

4. pip3 install --upgrade pip (In Mac or Linux)
   pip install --upgrade pip (In Windows)
   
4. pip3 install -r requirements.txt (In Mac or Linux)
   pip install -r requirements.txt (In Windows)
```

> Formatter Used - black

### Project Structure
```
.
├── README.md
├── dataloader
│   └── data.py
├── eval.py
├── logs
├── metrics.py
├── models
│   └── unet.py
├── predict.py
├── requirements.txt
├── results
├── saved_models
└── train.py
```

Thank You!