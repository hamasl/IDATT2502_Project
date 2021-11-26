# IDATT2502_Project

By:
* [diderikk](https://github.com/diderikk)
* [RokasBliu](https://github.com/RokasBliu)
* [hamasl](https://github.com/hamasl)
## Description
The task was to create a machine learning model with the ability to detect software vulnerabilities in source code.
C was selected as the language to analyze.

## Important note
This project was created on Linux and MacOs therefore it has not been tested on Windows and may not work there.

## Requirements
* [python 3.8 (or later)](https://www.python.org)
* sh compatible shell e.g. [bash shell (or similar)](https://www.gnu.org/software/bash/)
* [ClangFormat](https://clang.llvm.org/docs/ClangFormat.html)

## Recommendations
* CUDA (for fast runtime)

## Packages
* [matplotlib](https://matplotlib.org)
* [torch](https://pytorch.org)
* [PyYAML](https://pyyaml.org)
* [numpy](https://numpy.org)
* [tqdm](https://tqdm.github.io)
* [scikit-learn](https://scikit-learn.org/stable/index.html)
* [sklearn](https://sklearn.org)

## Install packages
Use command:
```
pip install -r requirements.txt
```

## Folder structure
```
project
|   .github
|   app
|---data
|   |---CWE835_Infinite_Loop
|   |   |   CWE835_Infinite_Loop__while_01.c
|   |   |   CWE835_Infinite_Loop__while_02.c
|   |   |   ...
|   |
|   |---CWE369_Divide_by_Zero
|   |   |   CWE369_Divide_by_Zero__float_connect_socket_01.c
|   |   |   CWE369_Divide_by_Zero__float_connect_socket_02.c
|   |   |   ...
|   |   ...
|   |
|---formatted
|   |---CWE835_Infinite_Loop
|   |   |   CWE835_Infinite_Loop__while_01.c.formatted
|   |   |   CWE835_Infinite_Loop__while_02.c.formatted
|   |   |   ...
|   |
|   |---CWE369_Divide_by_Zero
|   |   |   CWE369_Divide_by_Zero__float_connect_socket_01.c.formatted
|   |   |   CWE369_Divide_by_Zero__float_connect_socket_02.c.formatted
|   |   |   ...
|   |   ...
|   |
|---model
|   |---state
|   |   |   cnn_state.pth
|   |   |   hyper_params.txt
|   |   |
|   |   __main__.py
|   |   cnn.py
|   |   ...
|   |
|   preprocessing
|---processed
|   |   x.pt
|   |   y.pt
|   |
|   scripts
|---test
|   |   model
|   |   preprocessing
|   .gitignore
|   Makefile
|   README.md
|   requirements.txt
```

## How to run
### Format training data
The training data firstly needs to be put in a folder called data where there should be a directory per vulnerability as shown in the folder structure.
Formatting is necessary for the preprocessing to be able to correctly tokenize the source code.  
With make:
```
make format_test_files
```

Without make:
```
./scripts/format_test_files.sh
```

### Preprocess training data
To preprocess the training data the data needs to be formatted from the previous step.
To verify this check that the formatted folder is as shown in the folder structure section.

With make:
```
make run_preprocessing
```

Without make:
```
mkdir -p processed
mkdir -p preprocessing/plots
mkdir -p ./preprocessing/state
python3 -m preprocessing
```

### Train model
When preprocessing is done the model can be trained with the data that should be stored in x.pt and y.pt under the processed folder as seen in the folder architecture.  
With make:
```
make run_model
```

Without make:
```
mkdir -p model/plots
mkdir -p ./model/state
python3 -m model
```


### Analyze source code
After training the model there should be a cnn_state.pth and hyper_params.txt saved under model/state, these are used when predicting new code.  
With make:
```
make run_predict file_path="..."
```
If the file is located at ./vulnerable_files/example.c, then the command should look like:
```
make run_predict file_path="vulnerable_files/example.c"
```

Without make:
```
clang-format $(file_path) > $(file_path).formatted
python3 -m app $(file_path).formatted
rm -f $(file_path).formatted
```
If the file is located at ./vulnerable_files/example.c, then the command should look like:
```
clang-format vulnerable_files/example.c > vulnerable_files/example.c.formatted
python3 -m app vulnerable_files/example.c.formatted
rm -f vulnerable_files/example.c.formatted
```