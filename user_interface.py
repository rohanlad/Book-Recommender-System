import pandas as pd
import os
from cars_books import train_models_cars, load_models_cars
from hybrid_books import train_models_hybrid, load_models_hybrid
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

ratings_data = pd.read_csv('cleaned_book_ratings.csv', index_col=0)

while True:
    try:
        active_user_id = int(input("\nPlease enter your user_id to begin: "))
    except ValueError:
        print('\nSorry, that is not a valid user_id!')
        continue
    if active_user_id not in ratings_data['user_id'].unique():
        print('\nSorry, we do not recognise that user_id!\n')
        continue
    active_user_age = ratings_data.loc[ratings_data['user_id'] == active_user_id, 'age'].iloc[0]
    while True:
        desired_option = input('\n Please enter the number corresponding to your desired action:\n (1) Get some book recommendations\n (2) Submit a rating\n (3) View/Update your profile \n (4) Read our data collection notice\n (5) Log out\n')

        if desired_option == '1':
            while True:
                desired_recommender = input('\nWe have two different frameworks that are capable of generating recommended books for you. The first one is known as a Context Aware Recommender System (CARS), which will take into account your contextual data in order to help generate the best recommendations for you. The second one is known as a Hybrid Recommender System, which uses additional features about the books themselves to help generate the best recommendations. Which option would you like?\n (1) CARS\n (2) Hybrid\n ')
                if desired_recommender == '1':
                    print('\nGetting recommendations, please wait one moment.\n')
                    recommendations = load_models_cars(ratings_data, active_user_id, active_user_age, 15)
                    for key, value in recommendations.items():
                        title = ratings_data.loc[ratings_data['isbn'] == key, 'book_title'].iloc[0]
                        if pd.isnull(title) or title == '':
                            title = key
                        print(f"{title}: {value}")
                    break
                elif desired_recommender == '2':
                    print('\nGetting recommendations, please wait one moment.\n')
                    recommendations = load_models_hybrid(ratings_data, active_user_id, 15)
                    for key, value in recommendations.items():
                        title = ratings_data.loc[ratings_data['isbn'] == key, 'book_title'].iloc[0]
                        if pd.isnull(title) or title == '':
                            title = key
                        print(f"{title}: {value}")
                    break
                else:
                    print('\nInvalid option given - please input either 1 or 2')
                    continue

        elif desired_option == '2':
            isbn = input('\nPlease enter the ISBN number of the book you wish to submit a rating for: ')
            book_author = input('\nPlease enter the author of the book: ')
            while True:
                try:
                    rating = int(input('\nPlease enter your rating of the book (a number from 1 to 10): '))
                    if rating < 1 or rating > 10:
                        print('\nError: Please enter a rating between 1 and 10!')
                        continue
                except ValueError:
                    print('\nSorry, that is not a valid rating!')
                    continue
                try:
                    publication_year = int(input('\nPlease enter the year the book was published: '))
                except ValueError:
                    print('\nSorry. that is not a valid year!')
                    continue 
                ratings_data.loc[len(ratings_data)] = {'user_id': active_user_id, 'isbn': isbn, 'book_rating': rating, 'book_title': isbn, 'book_author': book_author, 'year_of_publication': publication_year, 'age': active_user_age}
                print('\nThanks for submitting your rating. We will now need to re-train our recommender systems to adapt to this new data. This will likely take a very long time so please come back to the system later to get recommendations again.\n')
                train_models_cars(ratings_data)
                train_models_hybrid(ratings_data)
                ratings_data.to_csv('cleaned_book_ratings.csv')
                break

        elif desired_option == '3':
            pass
            print('\nYour user_id is: ' + str(active_user_id))
            print('Your age is: ' + str(active_user_age))
            while True:
                desired_profile_action = input('\n Please enter the number corresponding to your desired action:\n(1) Update your age\n(2) Return to the main menu\n')
                if desired_profile_action == '1':
                    try:
                        new_age = float(input('\n Please enter your age: '))
                        ratings_data.loc[ratings_data['user_id'] == active_user_id, 'age'] = new_age
                        print('\nThanks for updating your age. We will now need to re-train our recommender systems to adapt to this new data. This will likely take a long time so please come back to the system later to get recommendations again.\n')
                        train_models_cars(ratings_data)
                        train_models_hybrid(ratings_data)
                        ratings_data.to_csv('cleaned_book_ratings.csv')
                        break
                    except ValueError:
                        print('\nSorry, that is not a valid age!')
                        continue

                elif desired_profile_action == '2':
                    break
                else:
                    print('\nInvalid option given - please input either 1 or 2')
                    continue
            
        elif desired_option == '4':
            # Are users aware which data is collected, how and for which purposes?
            print('\nIn order to generate the best book recommendations for you, the system uses various aspects of your data to help it learn a more accurate model of your preferences and relationships to other users on the system. The purpose of this is so that we are able to generate the most accurate book recommendations for you. Note that you are anonymised in our system by a user_id. The only personal data that we store about you is your age, which our system uses as contextual data when determining which books to recommend. The value for this is available for you to view or update from this system at any time you wish. We collect and store any ratings that you have submitted for any books, and link these ratings to your user_id.')

        elif desired_option == '5':
            print('\n Goodbye for now!')
            break

        else:
            print('\nInvalid option given - please input an option from 1 to 5')
            continue
    