import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost.sklearn import XGBClassifier
from lightgbm.sklearn import LGBMClassifier
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')
pd.options.display.max_columns = None
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn import model_selection
import matplotlib


def main():
    train_data=pd.read_excel('trainset_t1c.xlsx', engine='openpyxl')
    test_data =pd.read_excel('testset_t1c.xlsx', engine='openpyxl')

    data = pd.concat([train_data,test_data],ignore_index = True)
    X = data.iloc[:,3:]
    y = data.iloc[:,2]
    y_id = data.iloc[:,0:3]
    for i in range(len(X.columns)):
        X.iloc[:,i] = pd.to_numeric(X.iloc[:,i],errors = "coerce")
    X = X.dropna(axis = 1)
    scaler = preprocessing.StandardScaler().fit(X)
    X_data_transformed = scaler.transform(X)
    X_data_transformed=pd.DataFrame(X_data_transformed)
    X_data_transformed.columns=X.columns
    X_data=X_data_transformed
    X = X_data

    # ############### Remove redundant #########################
    x_cols = [col for col in X.columns if X[col].dtype != 'object']
    labels = []
    values = []
    for col in x_cols:
        labels.append(col)
        values.append(abs(X[col].corr(y, 'spearman')))  # np.corrcoef(X[col].values,y.values)[0,1]
    corr_df = pd.DataFrame({'col_labels': labels, 'corr_values': values})
    features = corr_df['col_labels']
    feature_matrix = X[features]
    corr_matrix =feature_matrix.corr(method='spearman')#method='spearman'
    remove_features = []
    mask = (corr_matrix.iloc[:,:].values>0.9) & (corr_matrix.iloc[:,:].values<1)
    for idx_element in range(len(corr_matrix.columns)):
        for idy_element in range(len(corr_matrix.columns)):
            if mask[idx_element,idy_element]:
                if list(corr_df['corr_values'])[idx_element] > list(corr_df['corr_values'])[idy_element]:
                    remove_features.append(list(features)[idy_element])
                else:
                    remove_features.append(list(features)[idx_element])
    remove_features = set(remove_features)
    print(len(remove_features))

    remain_features = set(features) - remove_features
    Xremain = X.loc[:,remain_features]
    data1 = pd.concat([y_id,Xremain],axis=1)
    train_data = data1[data1['set'] == 1].iloc[:,2:]
    test_data = data1[data1['set'] == 0].iloc[:,2:]
    print('train shape:',train_data.shape)
    print('test shape:',test_data.shape)

    x_train = train_data.drop(["pathology"], axis=1)
    y_train = train_data["pathology"]
    x_test = test_data.drop(["pathology"], axis=1)
    y_test = test_data["pathology"]
    log_reg = LogisticRegression(solver= "sag")
    model = log_reg.fit(x_train,y_train)
    predict_train= log_reg.predict(x_test)
    print('Train AUC:',metrics.roc_auc_score(y_test, predict_train))

    models = [LogisticRegression(solver= "sag"),
              SVC(kernel="rbf",probability=True),
              DecisionTreeClassifier(),
              RandomForestClassifier(),
              GradientBoostingClassifier(),
              MLPClassifier(solver='lbfgs', max_iter=100),
              XGBClassifier(n_estimators = 100, objective='reg:squarederror'),
              LGBMClassifier(n_estimators = 50)]

    result = dict()
    for model in models:
        model_name = str(model).split('(')[0]
        scores = cross_val_score(model, X=x_train, y=y_train, verbose=0, cv = 5, scoring=make_scorer(metrics.accuracy_score))
        result[model_name] = scores
        print(model_name + ' is finished')

    result = pd.DataFrame(result)
    result.index = ['cv' + str(x) for x in range(1, 6)]
    result

    matplotlib.rcParams['figure.figsize'] = (18.0, 18.0)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    i = 0
    plt.xticks(np.arange(0, 1.1, step=0.1))
    plt.yticks(np.arange(0, 1.1, step=0.1))
    result = dict()
    cv = model_selection.ShuffleSplit(n_splits=3, test_size=0.3, random_state=1)
    cs = ['red', 'orange', 'yellow', 'green', 'cyan',
          'blue', 'purple', 'pink', 'magenta', 'brown']
    c = 0
    for model in models:
        model_name = str(model).split('(')[0]
        for train, test in cv.split(x_train, y_train):
            probas_ = model.fit(x_train.iloc[train], y_train.iloc[train]).predict_proba(x_train.iloc[test])
            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = roc_curve(y_train.iloc[test], probas_[:, 1])
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            i += 1

        plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='gray', alpha=.8)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color=cs[c], label=model_name + r' ROC AUC = %0.2f $\pm$ %0.2f' % (mean_auc, std_auc),
                 lw=2, alpha=.8)
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        c += 1

        plt.xlabel('1-Specificity', fontsize='x-large')
        plt.ylabel('Sensitivity', fontsize='x-large')

        plt.legend(loc="lower right", prop={"size": 22})
    # plt.savefig('Train-ROC.jpg',dpi=300)

    plt.show()


    #test
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    lw=2
    i = 0
    matplotlib.rcParams['figure.figsize'] = (18.0, 18.0)

    plt.xticks(np.arange(0, 1.1, step=0.1))
    plt.yticks(np.arange(0, 1.1, step=0.1))
    result = dict()
    cv = model_selection.ShuffleSplit(n_splits = 3, test_size = 0.3, random_state = 1)
    cs = ['red','orange','yellow','green','cyan',
          'blue','purple','pink','magenta','brown']
    c = 0
    for model in models:
        model_name = str(model).split('(')[0]
        for train, test in cv.split(x_train, y_train):
            trainporbas_= model.fit(x_train.iloc[train], y_train.iloc[train]).predict_proba(x_train.iloc[test])
            probas_ = model.predict_proba(x_test)  #需要修改的是clf，即训练得到的model；以及测试集的X_test和y_test.
            fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
            fpr=fpr
            tpr=tpr
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
        plt.plot(fpr, tpr, color=cs[c], alpha=.8, lw=lw, linestyle='-',label= model_name + r' ROC AUC = %0.2f' % roc_auc)
        plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='gray', alpha=.8)

        plt.xlabel('1-Specificity', fontsize = 'x-large')
        plt.ylabel('Sensitivity', fontsize = 'x-large')
            #plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right" ,prop={"size":22})
        c +=1
            #plt.savefig('Test-ROC.tiff',dpi=200)
    plt.show()

if __name__ == '__main__':
    main()
