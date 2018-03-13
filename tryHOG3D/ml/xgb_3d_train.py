# -------------------------  Refs --------------------------------
# [Xgboost python API Manual](http://xgboost.readthedocs.io/en/latest/python/python_api.html)
# [Xgboost python usage introduction](http://xgboost.readthedocs.io/en/latest/python/python_intro.html)

import time
import cv2 as cv
import numpy as np
import xgboost as xgb

def logregobj(preds, dtrain):
    labels = dtrain.get_label()
    preds = 1.0 / (1.0 + np.exp(-preds))
    grad = preds - labels
    hess = preds * (1.0 - preds)
    return grad, hess

def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    # return a pair metric_name, result
    # since preds are margin(before logistic transformation, cutoff at 0)
    return 'error', float(sum(labels != (preds >= 0.5))) / len(labels)

t = time.time()
TRAIN_NUM = "1"
TEST_NUM = "1"
dtrain = xgb.DMatrix("../features3d/feature3d_%s.txt"%TRAIN_NUM)
dtest = xgb.DMatrix("../features3d/feature3d_%s.txt"%TEST_NUM)

labels = dtrain.get_label()
print(labels)

# param is a dictionary. you can refer to xgboost python intro for further info on its keys and available values. Its obj-func and eval can be defined by users
param = {'max_depth': 20, 'eta': 1, 'silent': 0, 'subsample': 0, 'objective':'binary:logistic' }


# param = {'max_depth': 10, 'eta': 1, 'silent': 0, 'subsample': 0 }

watchlist = [(dtest, 'eval'), (dtrain, 'train')]
num_round = 20

# dst = xgb.train(param, dtrain, num_round, watchlist, logregobj, evalerror) # dst is xgb.Booster. 
dst = xgb.train(param, dtrain, num_round, watchlist)

dst.save_model("./hog3d.model")