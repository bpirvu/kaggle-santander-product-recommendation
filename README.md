# Kaggle Competition: Santander Product Recommendation

This repository reflects the source code of our submission to the Santander Product Recommendation Kaggle Competition that initially brought us to the 25th place.

After a break of around 3 weeks during which we didn't make any submissions we dropped to some place around 120. Shortly before the competition ended we fine-tuned our submission and we ended up on the 88th place out of 1785 participants. However the fine-tuned code is a pretty big mess which would take a while to clean up before making it public, this is why we decided to publish only the code for the initial submission, which is relatively clean.

We used Python and GraphLab Create for data munging and model training. You can get an academic license of GraphLab Create at https://turi.com/download/academic.html.

Generally speaking, we tried to keep our approach as simple as possible. Here are its key ideas:

* no data cleaning (we simply dropped all rows with invalid data)
* model used: the Boosted Trees (Multiclass) Classifier from GraphLab Create
* use the transition 2015-05 -> 2015-06  in order to train the model
* use the product features from January to May 2015 in order to generate features (lookback_months=4)
* remove following 5 irrelevant product columns from the training data and from the predictions according to (https://www.kaggle.com/c/santander-product-recommendation/forums/t/25727/question-about-map-7?forumMessageId=146330#post146330):
    * irrelevantProductCols =['ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_viv_fin_ult1','ind_deco_fin_ult1','ind_deme_fin_ult1']
    
One more remark: GraphLab Create is evaluatin many operations lazily, which yields increasing computational times in various loops. In order to avoid this issue we persisted the SFrames during the computation at each iteration. This may look a bit odd, however it improves the performance considerably.

Many thanks to the Kaggle forum users and the Vienna Kaggle meetup for inspiring ideas!
