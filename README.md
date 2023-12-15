# IFT6390B - Data Challenge #2

## Contents

- `randomforest.ipynb`: This notebook was used to develop the random forest code and a few helper functions to handle the submission
- `run_models.py`: This file trains or loads the different types of models we prepared, then saves their predictions, properly formatted, into .csv files.
- `sign_language_v0.0.3.ipynb`: This notebook was used to develop the CNN model. It is not intended to be run from beginning to end and is not to be considered like a "finished product"; we include it only to show our work.
- `functions/`: the Python library we created to collect the varied functionality we created in one place
    - `__init__.py`: allows the import module to understand the directory as a library; contains general-purpose and input/output tools.
    - `cnn.py`: collects our specific CNN design into a class that can be interchanged with other model classes.
    - `rf.py`: random-forest functionality