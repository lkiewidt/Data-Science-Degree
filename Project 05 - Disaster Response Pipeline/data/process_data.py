import sys
import pandas as pd
import sqlalchemy as db


def load_data(messages_filepath, categories_filepath):
    """
    Load and combine datasets into a DataFrame
    
    Inputs
    ------
    messages_filepath : str
        path to messages dataset (must be .csv)
        
    categories_filepath : str
        path to categories dataset (must be .csv)
        
    Returns
    -------
    df : DataFrame
        DataFrame combining the messages and categories
    """
    
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = pd.merge(messages, categories, on='id')
    
    return df


def clean_data(df):
    """
    Clean disaster message dataset:
    - extract and assign category names
    - one-hot encode category values
    - remove duplicate entries
    
    Inputs
    ------
    df : DataFrame
        original uncleanded dataframe
    
    Returns
    -------
    df : DataFrame
        cleaned dataframe
    """
    
    # create a dataframe of the individual category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # extract category names from first row
    row = categories.loc[0]
    category_colnames = row.apply(lambda s: s[:-2]).to_list()
    
    # rename category columns
    categories.columns = category_colnames
    
    # one-hot encode category data (via transformation of strings)
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
    
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column], downcast='integer')
    
    # drop the original categories column from `df`
    df = df.drop('categories', axis=1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # drop duplicate entries
    df.drop_duplicates(inplace=True)
    
    return df


def save_data(df, database_filename):
    """
    Saves data from DataFrame into sqlite database.
    
    Inputs
    ------
    df : DataFrame
        DataFrame to be saved

    database_filename : str
        name of the databased to be saved
    """
    
    # save cleaned data into database
    engine = db.create_engine('sqlite:///' + database_filename)
    df.to_sql('cleaned_messages', engine, index=False)
    
    return 0  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()