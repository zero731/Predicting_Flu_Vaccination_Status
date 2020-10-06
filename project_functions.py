import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics


def check_null(df):
    import pandas as pd
    missing_vals = pd.DataFrame()
    missing_vals['Number of Nulls'] = df.isna().sum()
    missing_vals['% Null'] = (df.isna().sum() / len(df)) * 100
    return missing_vals
    
    
    
    
    
    

def check_unique(df, col, dropna=False):
    import pandas as pd
    if dropna:
        unique_vals = pd.DataFrame(df[col].value_counts())
    else:
        unique_vals = pd.DataFrame(df[col].value_counts(dropna=False))
    return unique_vals









def check_col_distr(df, col, figsize=(7,5)):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    ## check counts of unique values in col
    display(check_unique(df, col))

    ## plot distribution of col
    plt.figure(figsize=figsize)
    ax = sns.distplot(df[col])
    return ax








def plot_box(feature, data, target='seasonal_vaccine'):
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(7,5))
    ax = sns.boxplot(x=target, 
                     y=feature, 
                     data=data, 
                     palette='nipy_spectral');
    ax.set_title('Vaccinated vs {}'.format(feature), fontsize=16, weight='bold')
    ax.set_xlabel('Vaccine', fontsize=14, weight='bold')
    ax.set_ylabel(feature, fontsize=14, weight='bold')
    return ax









def plot_bar(feature, data, target='seasonal_vaccine', hue='seasonal_vaccine', show_legend=False):
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(7,5))
    ax = sns.barplot(x=target,
                     y=feature,
                     palette='nipy_spectral',
                     hue=hue,
                     data=data)
    ax.set_title('Vaccinated vs {}'.format(feature), fontsize=16, weight='bold')
    ax.set_xlabel('Vaccine', fontsize=14, weight='bold')
    ax.set_ylabel(feature, fontsize=14, weight='bold')
    
    if show_legend==False:
        ax.get_legend().remove()
    
    return ax








def plot_reg(feature, data, category=None, target='seasonal_vaccine'):
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(7,5))
    g = sns.lmplot(x=target,
                     y=feature,
                     palette='nipy_spectral',
                     hue=category,
                     data=data,
                     scatter_kws={'alpha':0.5})
    g.set_axis_labels('Vaccine', feature) 
    return g







def plot_bb(feature, data, target='seasonal_vaccine'):
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14,6))

    sns.boxplot(x=target, 
                     y=feature, 
                     data=data, 
                     palette='nipy_spectral',
                     ax=ax1);
    ax1.set_title('Vaccinated vs {}'.format(feature), fontsize=16, weight='bold')
    ax1.set_xlabel('Vaccine', fontsize=14, weight='bold')
    ax1.set_ylabel(feature, fontsize=14, weight='bold')
    
    sns.barplot(x=target,
                     y=feature,
                     palette='nipy_spectral',
                     hue=target,
                     data=data,
                     ax=ax2)
    ax2.set_title('Vaccinated vs {}'.format(feature), fontsize=16, weight='bold')
    ax2.set_xlabel('Vaccine', fontsize=14, weight='bold')
    ax2.set_ylabel(feature, fontsize=14, weight='bold')
    ax2.get_legend().remove()
    
    plt.tight_layout()
    
    
    
    
    
    
    
    
def eval_classifier(clf, X_test, y_test, model_descr='',
                    target_labels=['No Vacc', 'Vaccine'],
                    cmap='Blues', normalize='true'):
    """Given an sklearn classification model (already fit to training data), test features, and test labels,
       displays sklearn.metrics classification report, confusion matrix, and ROC curve. A description of the model 
       can be provided to model_descr to customize the title of the classification report.
       """
    
    from sklearn.metrics import classification_report, plot_confusion_matrix, plot_roc_curve
    
    ## get model predictions
    y_hat_test = clf.predict(X_test)
    
    
    ## Classification Report
    report_title = 'Classification Report: {}'.format(model_descr)
    divider = ('-----' * 11) + ('-' * (len(model_descr) - 31))
    report_table = classification_report(y_test, y_hat_test, 
                                                 target_names=target_labels)
    
    print(divider, report_title, divider, report_table, divider, divider, '\n', sep='\n')
    
    
    ## Make Subplots for Figures
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
    
    ## Confusion Matrix
    plot_confusion_matrix(clf, X_test, y_test, 
                                  display_labels=target_labels, 
                                  normalize=normalize, cmap=cmap, ax=axes[0])
    
    axes[0].set_title('Confusion Matrix', fontdict={'fontsize': 18,'fontweight': 'bold'})
    axes[0].set_xlabel(axes[0].get_xlabel(),
                       fontdict={'fontsize': 12,'fontweight': 'bold'})
    axes[0].set_ylabel(axes[0].get_ylabel(),
                       fontdict={'fontsize': 12,'fontweight': 'bold'})
    axes[0].set_xticklabels(axes[0].get_xticklabels(),
                       fontdict={'fontsize': 10,'fontweight': 'bold'})
    axes[0].set_yticklabels(axes[0].get_yticklabels(), 
                       fontdict={'fontsize': 10,'fontweight': 'bold'})
    
    
    ## ROC Curve
    plot_roc_curve(clf, X_test, y_test, ax=axes[1])
    # plot line that demonstrates probable success when randomly guessing labels
    axes[1].plot([0,1],[0,1], ls='--', color='r')
    
    axes[1].set_title('ROC Curve', 
                      fontdict={'fontsize': 18,'fontweight': 'bold'})
    axes[1].set_xlabel(axes[1].get_xlabel(), 
                      fontdict={'fontsize': 12,'fontweight': 'bold'})
    axes[1].set_ylabel(axes[1].get_ylabel(), 
                      fontdict={'fontsize': 12,'fontweight': 'bold'})
    
    fig.tight_layout()
    plt.show()

    
    
    
    
    
    
def fit_grid_clf(model, params, X_train, y_train, score='accuracy'):
    
    from sklearn.model_selection import GridSearchCV
    
    grid = GridSearchCV(model, params, scoring=score, cv=3, n_jobs=-1)

    grid.fit(X_train, y_train)
    return grid
    
    
    
    
    
    
    
    
def plot_logreg_coeffs(model, feature_names, model_step='logreg',
                       title='Logistic Regression Coefficients'):

    logreg_coeffs = model.named_steps[model_step].coef_
    sorted_idx = logreg_coeffs.argsort()

    importance = pd.Series(logreg_coeffs[0], index=feature_names)
    fig, axes = plt.subplots(2, figsize=(12,10))
    importance.sort_values().tail(20).plot(kind='barh', ax=axes[0])
    importance.sort_values().head(20).plot(kind='barh', ax=axes[1])
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    
    
    
    
    
    
def plot_feat_importance(clf, model_step_name, feature_names, model_title=''):

    feature_importances = (
        clf.named_steps[model_step_name].feature_importances_)

    sorted_idx = feature_importances.argsort()
    
    importance = pd.Series(feature_importances, index=feature_names)
    plt.figure(figsize=(12,10))
    fig = importance.sort_values().tail(20).plot(kind='barh')
    fig.set_title('{} Feature Importances'.format(model_title), fontsize=18, fontweight='bold')
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12)

    plt.show()

    
    
    
    
    
    
def plot_count_by_grp(group, data, hue='seasonal_vaccine',
                      labels=['No Vacc', 'Vaccine'], title='',
                      y_label='# of Respondents', x_label='',
                      x_tick_labels=False, rotate=True,
                      grp_order=None):
    
    
    font_dict = {}
    font_dict['title'] = {'fontsize':18, 'fontweight':'bold'}
    font_dict['ax_label'] = {'fontsize':14, 'fontweight':'bold'}
    font_dict['ticks'] = {'size':14}
    font_dict['legend'] = {'fontsize':12}
    
    plt.figure(figsize=(8,6))
    ax = sns.countplot(x=group, hue=hue,
                  data=data, palette='nipy_spectral',
                      order=grp_order)
    ax.set_title('Vaccination By {}'.format(title), fontdict=font_dict['title'])
    ax.set_xlabel(x_label, fontdict=font_dict['ax_label'])
    ax.set_ylabel(y_label, fontdict=font_dict['ax_label'])
    ax.tick_params(labelsize=font_dict['ticks']['size'])
    
    if rotate:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    if x_tick_labels:
        ax.set_xticklabels(x_tick_labels)

    ax.legend(labels=labels, fontsize=font_dict['legend']['fontsize'])
    plt.show();