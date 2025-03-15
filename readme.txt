From the command line, enter python3 user_interface.py in order to launch the interactive user interface, from where recommendations can be obtained.
The user interface is fairly straightforward to use but refer to the video for an overview of how to navigate it.
The user_ids can be found from the dataset (cleaned_book_ratings.csv) but an example to use would be 15608.
There is no need to run any of the other python files directly because the required functions are called from within user_interface.py

The cars_save_rank, cars_save_retrieve, hybrid_save_rank, hybrid_save_retrieve directories all contain the models' saved training data.
This saved data is then loaded in when recommendations are requested.