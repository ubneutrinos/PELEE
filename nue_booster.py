#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""@package nue_booster
Package to train boosted trees to isolate low-energy electron neutrinos

Takes as input searchingfornues TTrees.
"""

from operator import itemgetter

import pandas as pd
import xgboost as xgb
import shap

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, recall_score, precision_score, average_precision_score

labels = ["ncpi0", "cc", "ccpi0", "cosmic", "ext"]

titles = [
    r"$\nu$ NC $\pi^{0}$", r"$\nu_{\mu}$ CC", r"$\nu_{\mu}$ CC $\pi^{0}$",
    r"Cosmic", r"EXT"
]

bkg_queries = [
    "category==31", "category==2", "category==21", "category==4", "category==0"
]

variables = [
    "shr_dedx_Y", "shr_distance", "trk_chipr", "hits_y", "trk_distance", "pt", "trk_chimu",
    "is_signal", "shr_tkfit_dedx_Y", "shr_tkfit_dedx_U", "shr_tkfit_dedx_V", "p", "nu_e",
    "hits_ratio", "shr_dedx_U", "shr_dedx_V", "n_tracks_contained", "n_showers_contained",
    "shr_theta", "trk_len", "train_weight", "trk_score", "shr_score", "shr_energy_tot", "trk_energy_tot",
    "shr_phi", "trk_theta", "trk_phi", "tksh_angle", "tksh_distance", "CosmicIP", "shr_bragg_p", "shr_chipr",
    "shr_chimu", "trk_bragg_p", "shr_pca_2", "shr_pca_1", "shr_pca_0", "shr_bragg_mu", "trk_bragg_mu", "trk_pida",
    "topological_score", "slpdg"
]

class NueBooster:
    """Main NueBooster class

    Args:
        samples (dict): Dictionary of pandas dataframes.
            mc`, `nue`, and `ext` are required.
        training_vars (list): List of variables used for training.
        random_state: seed for splitting sample. Default is 0.

    Attributes:
       samples (dict): Dictionary of pandas dataframes.
       random_state: seed for splitting sample.
       variables (list): List of variables used for training.
       params (dict): XGBoost parameters.
    """

    def __init__(self, samples, training_vars, random_state=0):
        self.samples = samples
        self.random_state = random_state
        self.variables = training_vars

        eta = 0.1
        max_depth = 10
        subsample = 1
        colsample_bytree = 1
        min_child_weight = 1
        self.params = {
            "objective": "binary:logistic",
            "booster": "gbtree",
            "eval_metric": "auc",
            "eta": eta,
            "tree_method": 'exact',
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "silent": 1,
            "min_child_weight": min_child_weight,
            "seed": random_state,
            #"num_class" : 22,
        }
        print(
            'XGBoost params. ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'
            .format(eta, max_depth, subsample, colsample_bytree))


    def _run_single(self, train, test, features, target, ax, title=''):
        num_boost_round = 1000
        early_stopping_rounds = 50

        y_train = train[target]
        y_valid = test[target]
        dtrain = xgb.DMatrix(
            train[features], y_train, weight=train["train_weight"])
        dvalid = xgb.DMatrix(
            test[features], y_valid, weight=test["train_weight"])

        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        gbm = xgb.train(
            self.params,
            dtrain,
            num_boost_round,
            evals=watchlist,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False)

        print("Validating...")
        # check = gbm.predict(
        #     xgb.DMatrix(test[features]), ntree_limit=gbm.best_iteration + 1)

        #area under the precision-recall curve
        # score = average_precision_score(test[target].values, check)
        # print('area under the precision-recall curve: {:.6f}'.format(score))

        # check2 = check.round()
        # score = precision_score(test[target].values, check2)
        # print('precision score: {:.6f}'.format(score))

        # score = recall_score(test[target].values, check2)
        # print('recall score: {:.6f}'.format(score))

        imp = self.get_importance(gbm, features)
        # print('Importance array: ', imp)

        ############################################ ROC Curve

        # Compute micro-average ROC curve and ROC area
        # fpr, tpr, _ = roc_curve(test[target].values, check)
        # roc_auc = auc(fpr, tpr)
        # xgb.plot_importance(gbm)
        # explainer = shap.TreeExplainer(gbm)
        # shap_values = explainer.shap_values(train[features])
        # shap.force_plot(explainer.expected_value, shap_values, train[features])
        # shap.summary_plot(shap_values, train[features], max_display=10)

        # lw = 2
        # ax.plot(fpr, tpr, lw=lw, label='%s (area = %0.2f)' % (title, roc_auc))
        # ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

        return gbm, imp, gbm.best_iteration + 1


    def get_importance(self, gbm, features):
        self.create_feature_map(features)
        importance = gbm.get_fscore(fmap='pickles/xgb.fmap')
        importance = sorted(importance.items(), key=itemgetter(1), reverse=True)
        return importance


    def train_booster(self, ax, bkg_query=""):

        plt_title = 'Global'

        if bkg_query in bkg_queries:
            print("Training %s..." % labels[bkg_queries.index(bkg_query)])
            plt_title = r"%s background" % titles[bkg_queries.index(bkg_query)]
            bkg_query = "&" + bkg_query

        test_nue = self.samples["nue"][0].query("nu_e < 0.8 & trk_chipr > 0 & selected == 1 & category == 11")[self.variables]
        train_nue = self.samples["nue"][1].query("nu_e < 0.8 & trk_chipr > 0 & selected == 1 & category == 11")[self.variables]

        test_nc = self.samples["nc"][0].query("nu_e < 0.8 & trk_chipr > 0 & selected == 1")[self.variables]
        train_nc = self.samples["nc"][1].query("nu_e < 0.8 & trk_chipr > 0 & selected == 1")[self.variables]

        test_mc = self.samples["mc"][0].query("nu_e < 0.8 & trk_chipr > 0 & selected == 1" + bkg_query)[self.variables]
        train_mc = self.samples["mc"][1].query("nu_e < 0.8 & trk_chipr > 0 & selected == 1" + bkg_query)[self.variables]

        test_ext = self.samples["ext"][0].query("trk_chipr > 0 & selected == 1" + bkg_query)[self.variables]
        train_ext = self.samples["ext"][1].query("trk_chipr > 0 & selected == 1" + bkg_query)[self.variables]

        train = pd.concat([train_nue, train_mc, train_ext, train_nc])
        test = pd.concat([test_nue, test_mc, test_ext, test_nc])

        features = list(train.columns.values)
        features.remove('is_signal')
        features.remove('nu_e')
        features.remove('train_weight')

        preds, imp, num_boost_rounds = self._run_single(
            train,
            test,
            features,
            'is_signal',
            ax,
            title=plt_title)

        return preds

    @staticmethod
    def get_features(train):
        trainval = list(train.columns.values)
        output = trainval
        return sorted(output)

    @staticmethod
    def create_feature_map(features):
        outfile = open('pickles/xgb.fmap', 'w')
        for i, feat in enumerate(features):
            outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        outfile.close()
