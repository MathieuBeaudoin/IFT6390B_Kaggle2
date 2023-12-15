import numpy as np
import pandas as pd


# Script to import data while reducing memory usage, work by Guillaume Martin
# https://www.kaggle.com/code/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))    
    for col in df.columns:
        col_type = df[col].dtype        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df

def import_data(file):
    df = pd.read_csv(file, parse_dates=True, keep_date_col=True)
    df = reduce_mem_usage(df)
    return df

def split_off_labels(df, y_col="label"):
    _X = (
        df
        .drop(y_col, axis=1)
        .to_numpy()
        .astype(np.int16) # Downcast to save memory 
    )
    _y = df[y_col].to_numpy().astype(np.int8)
    return _X, _y

def to_image(array, label = True):
    array = np.array(array)
    start_idx = 1 if label else 0
    return array[start_idx:].reshape(28,28).astype(float)

def convert_to_char(ascii_sum):
    if ascii_sum > 122:
        return chr(ascii_sum - 65)
    return chr(ascii_sum)

def vectorized_char_converter(ascii_sums):
    ascii_sums[ascii_sums > 122] -= 65
    return np.vectorize(chr)(ascii_sums)

def convert_predictions_to_chars(predictions):
    return [str(convert_to_char(65 + pred)) for pred in predictions]

def vectorized_preds_to_chars(predictions):
    return vectorized_char_converter(predictions + 65).astype(str)

def wrangle_test_set(data):
    d = int(data.shape[1] / 2)
    return data.index, np.array([
        data.iloc[:, :d].to_numpy(),
        data.iloc[:, d:].to_numpy()
    ])

def adapt_preds_to_expectations(test_sets, model, index):
    ascii_values = np.array([model.predict(ts) for ts in test_sets])
    preds_as_chars = vectorized_preds_to_chars(ascii_values.sum(0).astype(int))
    if np.ndim(preds_as_chars) != 1:
        print(f"Shapes of\n ascii_values: {ascii_values.shape}\n ",
              f"preds_as_chars: {preds_as_chars.shape}")
        raise ValueError("Wrong shapes!")
    return pd.Series(
        preds_as_chars,
        index = index,
        name = "label"
    )

def check_accuracy(model, 
                   X_valid,
                   labels_valid,
                   approaches={
                       "Single-vote": True,
                       "Probabilistic-combination": False
                   },
                   verbose: bool = True,
                   **kwargs):
    accs = []
    for _type, fptp in approaches.items():
        preds = model.predict(
            X_valid, 
            first_past_the_post=fptp, 
            **kwargs
        )
        description = f"{_type} accuracy:"
        accs.append(acc := np.mean(preds == labels_valid))
        if verbose:
            print(f"{description.ljust(36)}{acc}")
    return accs

