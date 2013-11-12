This is a slightly modified version of my 2nd place entry to the Kaggle Stumbleup Evergreen competision.

== Overview ==

This model only uses the provided textual features. It extracts text from the html. It extracts boilerplate (additionally to the one provided) using boilerpipe.

Instead of training one classifer on each tag (h1, title, boilerplate, ...), I found it to work better emperically to use linear combinations of tags (e.g. 5 * h1 + title * ...). I did not manage to find a way to esitmate the parameters in way which prevents overfitting on this noisy dataset, therefore I went for a brute force approach, just calculate lot's of combination and throw them all in an ensemble. For this end, I implemented a simple parser to the name of the dataset.

I used the same brute force approach to preprocessing. I used stemming, tf-idf, lsi (svd), lda for preprocessing. 

The only classifier used is logisitic regression. Other classifiers (e.g. Random Forest), did not seem to improve the final result. No parameter search is preformed. This becomes less importaint / contra-productive when constructing ensembles.

I included two ways of producing the final ensemble. One using Least Squares for selecting the best model and then averaging them. The other is combiting all models using a Gradient Boosting Machine.

All results are cached, so e.g. when you add another classifier it should not take more then a few minutes.

== Changes compared to my actual submission ==

The mainly I removed all code that did not make it into final submission, which would have made it difficult to follow what this code does. I also removed some 'secret source', which only had a small contribution to the final score, but which I might want to reuse for another competision. Also the submission, which I selected, did include the GTB model.

== Getting it running ==

Install requirements

  pip install -r requirements.txt 

Place the competition data into data/ and unzip raw_content.zip into data/raw_content/

Extract text from html and create word count matrices.

  python src/parse_html.py && python src/count_words.py

This should not take more than 30 minutes. Now run the actual model.

  python src/train.py 
