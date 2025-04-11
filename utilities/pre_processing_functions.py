from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import pandas as pd

def encode_column_train_test(train_df, test_df, column_name, encoding_type, custom_order=None):
    """
    Encode a single column in both train and test sets.
    
    Args:
        train_df (DataFrame): Training DataFrame
        test_df (DataFrame): Test DataFrame
        column_name (str): Column to encode
        encoding_type (str): 'onehot' or 'ordinal'
        custom_order (list): For ordinal encoding, custom order of categories
    
    Returns:
        train_encoded_df, test_encoded_df (DataFrames)
    """
    train_col = train_df[[column_name]].copy()
    test_col = test_df[[column_name]].copy()

    if encoding_type == 'onehot':
        encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        train_encoded = encoder.fit_transform(train_col)
        test_encoded = encoder.transform(test_col)
        
        col_names = encoder.get_feature_names_out([column_name])
    
    elif encoding_type == 'ordinal':
        if custom_order:
            encoder = OrdinalEncoder(categories=[custom_order])
        else:
            encoder = OrdinalEncoder()
            
        train_encoded = encoder.fit_transform(train_col)
        test_encoded = encoder.transform(test_col)
        
        col_names = [f"{column_name}_ordinal"]
    
    else:
        raise ValueError("encoding_type must be either 'onehot' or 'ordinal'")
    
    train_encoded_df = pd.DataFrame(train_encoded, columns=col_names, index=train_df.index)
    test_encoded_df = pd.DataFrame(test_encoded, columns=col_names, index=test_df.index)

    return train_encoded_df, test_encoded_df
