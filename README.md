# Kaggle Competition: Santander Product Recommendation

This repository reflects the source code of our submission to the Santander Product Recommendation Kaggle Competition that initially brought us to the 25th place.

After a break of around 3 weeks during which we didn't make any submissions we dropped to some place around 120. Shortly before the competition ended we fine-tuned our submission and we ended up on the 88th place out of 1785 participants. However the fine-tuned code is a pretty big mess which would take a while to clean up before making it public, this is why we decided to publish only the code for the initial submission, which is relatively clean.

We used Python and GraphLab Create for data munging and model training. You can get an academic license of GraphLab Create at https://turi.com/download/academic.html.

Generally speaking, we tried to keep our approach as simple as possible. Here are its key ideas:

* no data cleaning (we simply dropped all rows with invalid data)
* model used: the Boosted Trees (Multiclass) Classifier from GraphLab Create
* use the transition 2015-05 -> 2015-06  in order to train the model
* use the product features from January to May 2015 in order to generate features (lookback_months=4)

Many thanks to the Kaggle forum users and the Vienna Kaggle meetup for inspiring ideas!
