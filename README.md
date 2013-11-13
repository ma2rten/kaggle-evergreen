This is a slightly modified version of my 2nd place entry to the Kaggle Stumbleupon Evergreen competition.

Overview
-----------

This model only uses textual features from the provided features. First I extract text from the html. I also extract 
boilerplate using boilerpipe (additionally to the one provided).

Instead of training one classifier on each tag (h1, title, boilerplate, ...), I found it to work better empirically 
to use linear combinations of tags (e.g. 5 * h1 + title * ...). I did not manage to find a way to estimate the 
tag weights in way which prevents overfitting on this noisy dataset, therefore I went for a brute force approach: 
just calculate lots of combination and throw them all in an ensemble. For this end, I implemented a simple parser, 
which parses the name of the dataset (e.g. 10 * title + body).

I used the same brute force approach to preprocessing, applying stemming, tf-idf, lsi (svd), lda to every dataset. This 
way I ended up with 260 models.

The only classifier used is logistic regression. Other classifiers (e.g. Random Forest), did not seem to improve the final result. Also no parameter search is preformed. 
After all, the best parameters per classifier are not necessarily the ones that improve the ensemble.

I included 3 ways of producing the final ensemble: simple average, weighted average using least squares and using least 
squares to select what models are averaged (this works best).

All results are cached, so e.g. when you add another classifier it should not take more then a few minutes.

Changes compared to my actual submission
-----------

I removed some 'secret source', which has to not been published yet. This code would give you a private leaderboard score of 0.88752 (or 6th place).

Getting it running
-----------

Install requirements

  ```pip install -r requirements.txt ```

Place the competition data into data/ and unzip raw_content.zip into data/raw_content/

Extract text from html and create word count matrices.

  ```python src/parse_html.py && python src/count_words.py```

This should not take more than 30 minutes. Now train the actual models:

  ```python src/train_models.py```
  
This takes about 120 minutes. Now you can get the final score with:

  ```python src/ensemble.py```
  
  That's it.
