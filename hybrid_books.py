# *********************
# CITATIONS
# *********************
# A lot of this code is adapted from the Tensorflow guides & tutorials. 
# https://www.tensorflow.org/recommenders/examples/basic_retrieval
# https://www.tensorflow.org/recommenders/examples/basic_ranking
# https://www.tensorflow.org/recommenders/examples/featurization
# https://www.tensorflow.org/recommenders/examples/context_features
# https://www.tensorflow.org/recommenders/examples/deep_recommenders

import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs

# The user model
class UserModel(tf.keras.Model):

  def __init__(self):
    super().__init__()

    self.user_embedding = tf.keras.Sequential([
        tf.keras.layers.IntegerLookup(
            vocabulary=unique_users, mask_token=None),
        tf.keras.layers.Embedding(len(unique_users) + 1, 64),
    ])
  def call(self, inputs):
    return self.user_embedding(inputs['user_id'])


# The book model
class BookModel(tf.keras.Model):

  def __init__(self):
    super().__init__()

    self.book_embedding = tf.keras.Sequential([
        tf.keras.layers.StringLookup(
            vocabulary=unique_books, mask_token=None),
        tf.keras.layers.Embedding(len(unique_books) + 1, 64)
    ])

    self.author_embedding = tf.keras.Sequential([
        tf.keras.layers.StringLookup(
            vocabulary=unique_authors, mask_token=None),
        tf.keras.layers.Embedding(len(unique_authors) + 1, 64)
    ])

    self.publication_year_embedding = tf.keras.Sequential([
        tf.keras.layers.Discretization(publication_year_buckets.tolist()),
        tf.keras.layers.Embedding(len(publication_year_buckets) + 1, 64),
    ])
    self.normalized_publication_year = tf.keras.layers.Normalization(
        axis=None
    )
    self.normalized_publication_year.adapt(publication_years)

  def call(self, inputs):
    return tf.concat([
        self.book_embedding(inputs['isbn']),
        self.author_embedding(inputs['book_author']),
        self.publication_year_embedding(inputs['year_of_publication']),
        tf.reshape(self.normalized_publication_year(
            inputs['year_of_publication']), (-1, 1))
    ], axis=1)


class QueryModel(tf.keras.Model):

  def __init__(self, layer_sizes):
    super().__init__()

    self.embedding_model = UserModel()

    self.dense_layers = tf.keras.Sequential()

    for layer_size in layer_sizes[:-1]:
      self.dense_layers.add(tf.keras.layers.Dense(
          layer_size, activation="relu"))

    for layer_size in layer_sizes[-1:]:
      self.dense_layers.add(tf.keras.layers.Dense(layer_size))

  def call(self, inputs):
    feature_embedding = self.embedding_model(inputs)
    return self.dense_layers(feature_embedding)


class CandidateModel(tf.keras.Model):
  
  def __init__(self, layer_sizes):
    super().__init__()

    self.embedding_model = BookModel()

    self.dense_layers = tf.keras.Sequential()

    for layer_size in layer_sizes[:-1]:
      self.dense_layers.add(tf.keras.layers.Dense(
          layer_size, activation="relu"))

    for layer_size in layer_sizes[-1:]:
      self.dense_layers.add(tf.keras.layers.Dense(layer_size))

  def call(self, inputs):
    feature_embedding = self.embedding_model(inputs)
    return self.dense_layers(feature_embedding)


class RetrieveMain(tfrs.models.Model):

  def __init__(self, layer_sizes):
    super().__init__()
    self.query_model = QueryModel(layer_sizes)
    self.candidate_model = CandidateModel(layer_sizes)
    self.task = tfrs.tasks.Retrieval(
        metrics=tfrs.metrics.FactorizedTopK(
            candidates=book_list.batch(128).map(self.candidate_model)
        ),
    )
  def compute_loss(self, features, training=False):
    # We pick out the user features and pass them into the user model, getting embeddings back
    query_embeddings = self.query_model({
        "user_id": features["user_id"]
    })
    # And pick out the book features and pass them into the book model,
    # getting embeddings back.
    book_embeddings = self.candidate_model({
        'isbn': features["isbn"],
        'book_author': features['book_author'],
        'year_of_publication': features['year_of_publication']
        })
    return self.task(
        query_embeddings, book_embeddings, compute_metrics=not training)


class RankingModel(tf.keras.Model):

  def __init__(self):
    super().__init__()

    # Compute embeddings for users.
    self.user_embeddings = tf.keras.Sequential([
        tf.keras.layers.IntegerLookup(
            vocabulary=unique_users, mask_token=None),
        tf.keras.layers.Embedding(len(unique_users) + 1, 64)
    ])

    # Compute embeddings for books.
    self.book_embeddings = tf.keras.Sequential([
        tf.keras.layers.StringLookup(
            vocabulary=unique_books, mask_token=None),
        tf.keras.layers.Embedding(len(unique_books) + 1, 64)
    ])

    # Compute predictions.
    self.ratings = tf.keras.Sequential([
        # Learn multiple dense layers.
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        # Make rating predictions in the final layer.
        tf.keras.layers.Dense(1)
    ])

  def call(self, inputs):
    user_id, isbn = inputs
    user_embedding = self.user_embeddings(user_id)
    book_embedding = self.book_embeddings(isbn)
    return self.ratings(tf.concat([user_embedding, book_embedding], axis=1))


class RankingMain(tfrs.models.Model):

  def __init__(self):
    super().__init__()
    self.ranking_model: tf.keras.Model = RankingModel()
    self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )

  def call(self, features):
    return self.ranking_model(
        (features["user_id"], features["isbn"]))

  def compute_loss(self, features, training=False):
    labels = features.pop("book_rating")
    rating_predictions = self(features)
    return self.task(labels=labels, predictions=rating_predictions)



def initial_setup(ratings_data):

  ratings_data['book_author'] = ratings_data['book_author'].fillna('')
  ratings_data['book_author'] = ratings_data['book_author'].astype('str')

  ratings_ratings = ratings_data[[
      'user_id', 'isbn', 'book_rating', 'book_author', 'year_of_publication']]
  ratings_books = ratings_data.drop_duplicates(
      'isbn')[['isbn', 'book_author', 'year_of_publication']]

  # convert dataframes to tensors
  tf_ratings_dict = tf.data.Dataset.from_tensor_slices(dict(ratings_ratings))
  tf_book_dict = tf.data.Dataset.from_tensor_slices(dict(ratings_books))

  global ratings
  ratings = tf_ratings_dict.map(lambda x: {
      'user_id': x['user_id'],
      'isbn': x['isbn'],
      'book_rating': x['book_rating'],
      'book_author': x['book_author'],
      'year_of_publication': x['year_of_publication']
  })
  global book_list
  book_list = tf_book_dict.map(lambda x: {
      'isbn': x['isbn'],
      'book_author': x['book_author'],
      'year_of_publication': x['year_of_publication']
  })

  global publication_years
  publication_years = np.concatenate(
      list(ratings.map(lambda x: x['year_of_publication']).batch(1000)))
  global publication_year_buckets
  publication_year_buckets = np.linspace(
      publication_years.min(), publication_years.max(), num=12,
  )

  # Create vocabularies
  global unique_users
  unique_users = np.unique(np.concatenate(
      list(ratings.map(lambda x: x['user_id']).batch(1000))))
  global unique_books
  unique_books = np.unique(np.concatenate(
      list(book_list.map(lambda x: x['isbn']).batch(1000))))
  global unique_authors
  unique_authors = np.unique(np.concatenate(
      list(ratings.map(lambda x: x['book_author']).batch(1000))))

  return ratings_data


def train_models_hybrid(_ratings_data_):
  ratings_data = initial_setup(_ratings_data_)
  train = ratings.take(int(len(ratings_data)*0.9))
  test = ratings.skip(int(len(ratings_data)*0.9)).take(int(len(ratings_data)*0.1))
  retrieve(train, test)
  rank(train, test)
  return


def retrieve(train, test):
  model = RetrieveMain([128, 64, 32])
  model.compile(optimizer=tf.keras.optimizers.Adagrad(0.01))
  model.fit(
      train.batch(8192),
      validation_data=test.batch(512),
      validation_freq=5,
      epochs=10,
      verbose=1)
  model.save_weights('hybrid_save_retrieve/')
  return


def rank(train, test):
  model_rank = RankingMain()
  model_rank.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.01))
  model_rank.fit(train.batch(8192), epochs=100)
  print('----------------------hybrideval')
  print(model_rank.evaluate(test.batch(8192), return_dict=True))
  print('----------------------hybrideval')
  model_rank.save_weights('hybrid_save_rank/')
  return


def load_models_hybrid(_ratings_data_, active_user_id, num_recommendations):
  ratings_data = initial_setup(_ratings_data_)
  model_retrieve_load = RetrieveMain([128, 64, 32])
  model_retrieve_load.load_weights('hybrid_save_retrieve/').expect_partial()
  index = tfrs.layers.factorized_top_k.BruteForce(
      model_retrieve_load.query_model, k=num_recommendations)
  index.index_from_dataset(
      tf.data.Dataset.zip((book_list.map(lambda x: x['isbn']).batch(
          100), book_list.batch(100).map(model_retrieve_load.candidate_model)))
  )
  _, isbns = index({'user_id': tf.constant([active_user_id])})

  model_rank_load = RankingMain()
  model_rank_load.load_weights('hybrid_save_rank/').expect_partial()
  test_ratings = {}
  test_isbns = isbns[0].numpy().tolist()
  for _isbn in test_isbns:
      isbn = _isbn.decode("utf-8")
      test_ratings[isbn] = model_rank_load({
          "user_id": np.array([active_user_id]),
          "isbn": np.array([isbn])
      })

  recommendations = {}
  for isbn, score in sorted(test_ratings.items(), key=lambda x: x[1], reverse=True):
    recommendations[isbn] = score
  return recommendations


# This is the function that calculates the coverage metric.
# It is not currently called anywhere, and should be called
# seperately in order to carry out the metric calculation. 
def hybrid_coverage(_ratings_data_, sampled_users):
    ratings_data = initial_setup(_ratings_data_)
    model_retrieve_load = RetrieveMain([128, 64, 32])
    model_retrieve_load.load_weights('hybrid_save_retrieve/').expect_partial()
    model_rank_load = RankingMain()
    model_rank_load.load_weights('hybrid_save_rank/').expect_partial()
    index = tfrs.layers.factorized_top_k.BruteForce(model_retrieve_load.query_model, k=100)
    index.index_from_dataset(
    tf.data.Dataset.zip((book_list.map(lambda x: x['isbn']).batch(
          100), book_list.batch(100).map(model_retrieve_load.candidate_model)))
    )
    coverages = []
    for user in sampled_users:
      above_threshold_count = 0
      _, isbns = index({'user_id': tf.constant([user])})
      test_ratings = {}
      test_isbns = isbns[0].numpy().tolist()
      for _isbn in test_isbns:
          isbn = _isbn.decode("utf-8")
          test_ratings[isbn] = model_rank_load({
          "user_id": np.array([user]),
          "isbn": np.array([isbn])
      })
      recommendations = {}
      for isbn, score in sorted(test_ratings.items(), key=lambda x: x[1], reverse=True):
          recommendations[isbn] = score
      for key, value in recommendations.items():
        if value.numpy()[0][0] > 7:
          above_threshold_count += 1
      coverages.append((above_threshold_count/100)*100)
    return (sum(coverages)/len(coverages))
