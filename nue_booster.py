import pandas as pd
import xgboost as xgb
from operator import itemgetter

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
    "shr_dedx_Y", "shr_distance", "trk_chipr", "hits_y", "trk_distance", "pt",
    "is_signal", "shr_tkfit_dedx_Y", "shr_tkfit_dedx_U", "shr_tkfit_dedx_V", "p", "nu_e",
    "hits_ratio", "shr_dedx_U", "shr_dedx_V", "n_tracks_cc0pinp", "n_showers_cc0pinp",
    "shr_theta", "trk_len", "train_weight", "trk_score", "shr_score", "shr_energy_tot", "trk_energy_tot"
]

def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()


def get_importance(gbm, features):
    create_feature_map(features)
    importance = gbm.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=itemgetter(1), reverse=True)
    return importance


def get_features(train, test):
    trainval = list(train.columns.values)
    output = trainval
    return sorted(output)


def run_single(train, test, features, target, ax, title='', random_state=0):
    eta = 0.1
    max_depth = 10
    subsample = 1
    colsample_bytree = 1
    min_child_weight = 1

    print(
        'XGBoost params. ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'
        .format(eta, max_depth, subsample, colsample_bytree))
    params = {
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
    num_boost_round = 1000
    early_stopping_rounds = 50

    X_train, X_valid = train, test

    y_train = X_train[target]
    y_valid = X_valid[target]
    dtrain = xgb.DMatrix(
        X_train[features], y_train, weight=X_train["train_weight"])
    dvalid = xgb.DMatrix(
        X_valid[features], y_valid, weight=X_valid["train_weight"])

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    gbm = xgb.train(
        params,
        dtrain,
        num_boost_round,
        evals=watchlist,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False)

    print("Validating...")
    check = gbm.predict(
        xgb.DMatrix(X_valid[features]), ntree_limit=gbm.best_iteration + 1)

    #area under the precision-recall curve
    score = average_precision_score(X_valid[target].values, check)
    print('area under the precision-recall curve: {:.6f}'.format(score))

    check2 = check.round()
    score = precision_score(X_valid[target].values, check2)
    print('precision score: {:.6f}'.format(score))

    score = recall_score(X_valid[target].values, check2)
    print('recall score: {:.6f}'.format(score))

    imp = get_importance(gbm, features)
    #     print('Importance array: ', imp)

    ############################################ ROC Curve

    # Compute micro-average ROC curve and ROC area
    fpr, tpr, _ = roc_curve(X_valid[target].values, check)
    roc_auc = auc(fpr, tpr)
    #xgb.plot_importance(gbm)
    #plt.show()

    lw = 2
    ax.plot(fpr, tpr, lw=lw, label='%s (area = %0.2f)' % (title, roc_auc))
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    return gbm, imp, gbm.best_iteration + 1


def train_booster(nue, mc, ext, variables, ax, bkg_query=""):
    print("Training %s..." % bkg_query)

    nue = nue.query("nu_e < 0.8 & selected == 1 & category == 11")
    filtered_nue = nue[variables]

    plt_title = 'Global'
    if bkg_query:
        plt_title = r"%s background" % titles[bkg_queries.index(bkg_query)]
        bkg_query = "&" + bkg_query

    mc = mc.query("nu_e < 0.8 & selected == 1" + bkg_query)
    filtered_mc = mc[variables]

    ext = ext.query("selected == 1" + bkg_query)
    filtered_ext = ext[variables]

    train_nue, test_nue = train_test_split(
        filtered_nue, test_size=0.5, random_state=1990)

    train = train_nue
    test = test_nue

    if len(mc):
        train_mc, test_mc = train_test_split(
            filtered_mc, test_size=0.5, random_state=1990)
        train = pd.concat([train, train_mc])
        test = pd.concat([test, test_mc])

    if len(ext):
        train_ext, test_ext = train_test_split(
            filtered_ext, test_size=0.5, random_state=1990)
        train = pd.concat([train, train_ext])
        test = pd.concat([test, test_ext])

    features = list(train.columns.values)
    features.remove('is_signal')
    features.remove('nu_e')
    features.remove('train_weight')

    preds, imp, num_boost_rounds = run_single(
        train,
        test,
        features,
        'is_signal',
        ax,
        title=plt_title,
        random_state=1990)

    print()

    return preds
