import pandas as pd
import numpy as np

# Note that this function is not currently called anywhere. It shows what we have done to process the raw dataset into
# the cleaned_book_ratings.csv dataset that is included in the submission. So if someone wanted to replicate that process,
# they would need to call this function with the raw dataset files in the current directory. Also note that this is just 
# the initial data processing and further processing is done in the respective files
def read_in_data():

    raw_ratings_data = pd.read_csv('BX-Book-Ratings.csv', encoding='latin-1', sep=";")
    raw_books_data = pd.read_csv('BX-Books.csv', encoding='latin-1', sep=";", on_bad_lines='skip')
    raw_user_data = pd.read_csv('BX-Users.csv', encoding='latin-1', sep=";")

    ratings_data = pd.merge(raw_ratings_data, raw_books_data, on='ISBN', how='left')
    ratings_data = pd.merge(ratings_data, raw_user_data, on='User-ID', how='left')

    ratings_data.columns = ratings_data.columns.str.replace('-', '_').str.lower()
    ratings_data.drop(columns=['image_url_s', 'image_url_m', 'image_url_l', 'publisher', 'location'], inplace=True)
    ratings_data['book_rating'] = pd.to_numeric(ratings_data['book_rating'], errors='coerce')
    ratings_data = ratings_data[ratings_data.book_rating > 0]
    ratings_data = ratings_data[ratings_data.book_rating < 11]
    ratings_data = ratings_data[ratings_data['book_rating'].notna()]
    ratings_data['user_id'] = pd.to_numeric(ratings_data['user_id'], errors='coerce')
    ratings_data = ratings_data[ratings_data['user_id'].notna()]
    ratings_data['isbn'] = ratings_data['isbn'].replace('', np.nan)
    ratings_data = ratings_data[ratings_data['isbn'].notna()]
    ratings_data['isbn'] = ratings_data['isbn'].astype('str') 
    ratings_data['isbn'] = ratings_data['isbn'].apply(lambda x: x.strip().strip("\'").strip("\\").strip('\"').strip("\#").strip("("))

    # ***** Relevant to CARS *****
    ratings_data['age'] = pd.to_numeric(ratings_data['age'], errors='coerce')
    ratings_data['age'] = ratings_data['age'].fillna(ratings_data['age'].mean())
    ratings_data = ratings_data[ratings_data.age < 120]
    ratings_data = ratings_data[ratings_data.age > 0]
    # ***** Relevant to CARS *****

    # ***** Relevant to Hybrid *****
    ratings_data['year_of_publication'] = pd.to_numeric(ratings_data['year_of_publication'], errors='coerce')
    ratings_data['year_of_publication'] = ratings_data['year_of_publication'].fillna(ratings_data['year_of_publication'].mean())
    
    # ***** Relevant to Hybrid *****
    # The vast majority of publication years fall within 1950 and 2005 so the following code results in minimal data loss and acts mainly
    # to remove anomalies e.g where the year is incorrect because it is in the future.
    ratings_data = ratings_data[ratings_data.year_of_publication > 1950]
    ratings_data = ratings_data[ratings_data.year_of_publication < 2005]
    # ***** Relevant to Hybrid *****

    # Shuffle the dataset
    ratings_data = ratings_data.sample(frac=1).reset_index(drop=True)
    ratings_data.to_csv('cleaned_book_ratings.csv')
