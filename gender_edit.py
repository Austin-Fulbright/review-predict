import pandas as pd
from tensorflow.keras.models import load_model

# Load the model
model = load_model('boyorgirl_5.h5')

import numpy as np

def preprocess(names_df, column_name='critic_name'):
    # Drop rows where name is not a string
    names_df = names_df[names_df[column_name].apply(lambda x: isinstance(x, str))]

    # Step 1: Lowercase
    names_df[column_name] = names_df[column_name].str.lower()

    # Step 2: Split individual characters
    names_df[column_name] = [list(name) for name in names_df[column_name]]

    # Step 3: Pad names with spaces to make all names same length
    name_length = 50
    names_df[column_name] = [
        (name + [' ']*name_length)[:name_length]
        for name in names_df[column_name]
    ]

    # Step 4: Encode Characters to Numbers
    names_df[column_name] = [
        [
            max(0.0, ord(char)-96.0)
            for char in name
        ]
        for name in names_df[column_name]
    ]

    # Convert lists to 2D NumPy array
    names = np.asarray(names_df[column_name].tolist())

    # Reshape the array to match the input shape that the model is expecting
    names = names.reshape(1, -1)

    predictions = model.predict(names)
    return names_df, predictions


def predict_gender(df):
    # Preprocess the names
    # This will depend on how you preprocessed names during training
    print("preprocessing gender names...")
    names, predictions = preprocess(df)
    print("starting predictions for gender")
    # Make predictions
    

    # Convert predictions to 1 (boy) or 0 (girl)
    # This will depend on how your model outputs predictions
    print("genders label binary created...")
    genders = [1 if pred < 0.5 else 0 for pred in predictions]

    # Replace names with predictions
    names['critic_name'] = genders

    return names
