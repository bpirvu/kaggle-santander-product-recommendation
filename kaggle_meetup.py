
# coding: utf-8

# In[2]:

import datetime
import dateutil
from os import getenv
import sys
import operator

import math
import numpy as np

DATADIR = getenv("DATADIR")
print DATADIR
#--------------------------------------------------
import graphlab as gl
from graphlab import SFrame
from graphlab import SArray
from graphlab import aggregate as agg
#--------------------------------------------------


# ---
# # Initial Data Preparation (execute only once)
# ## Import data & save as SFrames in order to improve efficiency

# In[1]:

test=SFrame("%s/KAGGLE/santander_product_recommendation/test_ver2.csv" % DATADIR)


# In[3]:

test.save('%s/KAGGLE/santander_product_recommendation/test_orig' % DATADIR)


# In[3]:

train=SFrame("%s/KAGGLE/santander_product_recommendation/train_ver2.csv" % DATADIR)


# In[5]:

train.save('%s/KAGGLE/santander_product_recommendation/train_orig' % DATADIR)


# In[6]:

# After executing previous step once, start directly here:
test_orig=SFrame("%s/KAGGLE/santander_product_recommendation/test_orig" % DATADIR)
train_orig=SFrame("%s/KAGGLE/santander_product_recommendation/train_orig" % DATADIR)


# In[7]:

renameDict = {
  'fecha_dato': 'Date',
  'ncodpers': 'PersonId',
  'ind_empleado': 'Employee',
  'pais_residencia': 'Country',
  'sexo': 'Sex',
  'age': 'Age',
  'fecha_alta': 'EntryDate',
  'ind_nuevo': 'New',
  'antiguedad': 'Seniority',
  'indrel': 'PrimaryCustomer',
  'ult_fec_cli_1t': 'LastDateAsPrimaryCustomer',
  'indrel_1mes': 'CustomerType',
  'tiprel_1mes': 'CustomerRelationType',
  'indresi': 'SameResidenceCountry',
  'indext': 'OtherBirthCountry',
  'conyuemp': 'EmployeeSpouse',
  'canal_entrada': 'EntryChannel',
  'indfall': 'Deceased',
  'tipodom': 'AddresType',
  'cod_prov': 'ProvinceCode',
  'nomprov': 'ProvinceName',
  'ind_actividad_cliente': 'Active',
  'renta': 'Income',
  'segmento': 'Segment'
}

train = train_orig
train.rename(renameDict)

test = test_orig
test.rename(renameDict)


# In[8]:

# There are 27734 rows that are totally broken:
# - all staticCols are missing or are an emtpy string.
# - only 'Date', 'PersonId' and the Product Columns contain proper entries.
def remove_broken_rows(data=SFrame()):
  good, bad = data.dropna_split('Active')
  return good


# In[9]:

train = remove_broken_rows(train)
train = train.sort(['PersonId','Date'])


# In[10]:

test.save('%s/KAGGLE/santander_product_recommendation/test' % DATADIR)
train.save('%s/KAGGLE/santander_product_recommendation/train' % DATADIR)


# ## After executing previous steps once, start directly here:

# In[13]:

staticCols = [
 'Employee',
 'Country',
 'Sex',
 'Age',
 'EntryDate',
 'New',
 'Seniority',
 'PrimaryCustomer',
 'LastDateAsPrimaryCustomer',
 'CustomerType',
 'CustomerRelationType',
 'SameResidenceCountry',
 'OtherBirthCountry',
 'EmployeeSpouse',
 'EntryChannel',
 'Deceased',
 'AddresType',
 'ProvinceCode',
 'ProvinceName',
 'Active',
 'Income',
 'Segment'
]

idCols = [
  'Date',
  'PersonId'
]

productCols = [
  'ind_cco_fin_ult1',
  'ind_cder_fin_ult1',
  'ind_cno_fin_ult1',
  'ind_ctju_fin_ult1',
  'ind_ctma_fin_ult1',
  'ind_ctop_fin_ult1',
  'ind_ctpp_fin_ult1',
  'ind_dela_fin_ult1',
  'ind_ecue_fin_ult1',
  'ind_fond_fin_ult1',
  'ind_hip_fin_ult1',
  'ind_nom_pens_ult1',
  'ind_nomina_ult1',
  'ind_plan_fin_ult1',
  'ind_pres_fin_ult1',
  'ind_reca_fin_ult1',
  'ind_recibo_ult1',
  'ind_tjcr_fin_ult1',
  'ind_valo_fin_ult1'
]

# https://www.kaggle.com/c/santander-product-recommendation/forums/t/25727/question-about-map-7?forumMessageId=146330#post146330
irrelevantProductCols = [
  'ind_ahor_fin_ult1',
  'ind_aval_fin_ult1',
  'ind_viv_fin_ult1',
  'ind_deco_fin_ult1',
  'ind_deme_fin_ult1'
]

allProductCols = productCols + irrelevantProductCols


# In[15]:

fNames = productCols
featureNames = list(set(fNames) - set(['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_viv_fin_ult1', 
                                                  'ind_deco_fin_ult1', 'ind_deme_fin_ult1']))
staticFeatureNames = staticCols + ['CalendarMonth']

def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def integerMonth (dateString):
    year = datetime.datetime.strptime(dateString, '%Y-%m-%d').year
    month = datetime.datetime.strptime(dateString, '%Y-%m-%d').month
    return (year-2015)*12+month


def delta(last, current):
    return (current > last and current == 1)


def concat_deltas(x):
    l = []
    for f in featureNames:
        if(x['Delta.'+f]):
            l.append(f)
    return l
  

def sort_dict_by_value(d):
    l = sorted(d.items(), key=operator.itemgetter(1), reverse=True)
    return list(i[0] for i in l)


def create7prediction(data, model):
    data = data.add_row_number()
    top7prediction = model.predict_topk(data, k=7).groupby('id', 
                                                {'Best7': agg.CONCAT('class', 'probability')})
    top7prediction['Sorted7'] = top7prediction['Best7'].apply(lambda x: sort_dict_by_value(x))
    data = data.join(top7prediction, 'id')
    return data


def build_groupby():
    d = {}
    for c in featureNames:
        d[c] = agg.CONCAT('DateInt', c)
    for c in staticFeatureNames:
        d[c] = agg.ARGMAX('DateInt', c)
    return d


def make_deltas(train, lookback_months):
    old_month = lookback_months
    new_month = lookback_months+1
    for column in featureNames:
        train['Delta.'+column] = train.apply(lambda x: delta(x[column+'.'+str(old_month)], x[column+'.'+str(new_month)]))
    train.save("%s/KAGGLE/santander_product_recommendation/tt" % DATADIR)
    train = SFrame("%s/KAGGLE/santander_product_recommendation/tt" % DATADIR)
    train['TotalDelta'] = train.apply(lambda x: concat_deltas(x))
    train['TotalDeltaString'] = train['TotalDelta'].apply(lambda x: ' '.join(x))
    train.save("%s/KAGGLE/santander_product_recommendation/tt" % DATADIR)
    train = SFrame("%s/KAGGLE/santander_product_recommendation/tt" % DATADIR)
    print 'before:', len(train)
    train = train.filter_by('', 'TotalDeltaString', exclude=True)
    print 'after:', len(train)
    train.save("%s/KAGGLE/santander_product_recommendation/tt" % DATADIR)
    train = SFrame("%s/KAGGLE/santander_product_recommendation/tt" % DATADIR)
    return train


def train_model(train, lookback_months):
    train = train.stack('TotalDelta', new_column_name='NewProduct')
    featureColumns = []
    targetColumns = []
    for f in featureNames:
        for i in range(lookback_months+1):
            featureColumns.append(f+'.'+str(i))
        targetColumns.append(f+'.'+str(lookback_months+1))

    staticColumns = list(set(staticFeatureNames)-set(['EmployeeSpouse', 'LastDateAsPrimaryCustomer',
                                                      'AddresType', 'CustomerType', 'CalendarMonth']))

    return gl.boosted_trees_classifier.create(train, verbose=True, max_iterations=20,
                        features=featureColumns+staticColumns, target='NewProduct')


def make_train_data(train, old_months, lookback_months):
    result = None
    for old_month in old_months:
        print old_month
        r = make_train_data_one_month(train, old_month, lookback_months)
        if result is None:
            result = r
        else:
            result = result.append(r)
    result.save("%s/KAGGLE/santander_product_recommendation/tt" % DATADIR)
    result = SFrame("%s/KAGGLE/santander_product_recommendation/tt" % DATADIR)
    return result
  
    
def make_train_data_one_month(t, target_old_month, lookback_months):
    t['DateInt'] = t['Date'].apply(lambda x: integerMonth(x))
    t['CalendarMonth'] = t['DateInt'].apply(lambda x: 'M'+str(x%12))
    t = t.filter_by(range(target_old_month-lookback_months, target_old_month+2), 'DateInt')
    t['DateInt'] = t['DateInt']-target_old_month+lookback_months
    d = build_groupby()
    t = t.groupby('PersonId', d)
    for c in featureNames:
        t = t.unpack(c)
    return t

  
def train_and_test(t, test, target_old_month, lookback_months):
    t = make_train_data(t, [target_old_month], lookback_months)
    return test.join(t, on=['PersonId'], how='inner')
  
    
def prepare_submission(model, lookback_months):
    t = SFrame("%s/KAGGLE/santander_product_recommendation/train" % DATADIR)
    test = SFrame("%s/KAGGLE/santander_product_recommendation/test" % DATADIR)
    ttest = train_and_test(t, test, 17, lookback_months)
    prediction = create7prediction(ttest, model)
    prediction['Sorted7String'] = prediction['Sorted7'].apply(lambda x: ' '.join(x))
    prediction = test.join(prediction, on='PersonId', how='left')
    return prediction
  

def test_model(model, target_old_month, lookback_months):
    t = SFrame("%s/KAGGLE/santander_product_recommendation/train" % DATADIR)
    t = make_train_data(t, [target_old_month], lookback_months)
    t = make_deltas(t, lookback_months)
    prediction = create7prediction(t, model)
    prediction['AveragePrecisionAt7'] = prediction.apply(lambda x: apk(x['TotalDelta'], x['Sorted7'], k=7))
    return prediction


# In[16]:

train = SFrame("%s/KAGGLE/santander_product_recommendation/train" % DATADIR)
lookback_months=4


# In[17]:

train = make_train_data(train, old_months=[4], lookback_months=lookback_months)


# In[18]:

train = make_deltas(train, lookback_months)


# In[19]:

model = train_model(train, lookback_months)


# In[22]:

submission = prepare_submission(model, lookback_months)


# In[23]:

submission.rename({'PersonId': 'ncodpers', 'Sorted7String': 'added_products'})
submission['ncodpers', 'added_products'].save("%s/KAGGLE/santander_product_recommendation/submission.csv" % DATADIR)


# In[ ]:



