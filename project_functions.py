import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics


def check_null(df):
    
    """ Takes in a Pandas DataFrame and returns a Pandas DataFrame displaying the number of null values 
    for each column in the original DataFrame, as well as the total percent of each column that is 
    made up of null values. 
    """
    
    import pandas as pd
    
    missing_vals = pd.DataFrame()
    missing_vals['Number of Nulls'] = df.isna().sum()
    missing_vals['% Null'] = (df.isna().sum() / len(df)) * 100
    
    return missing_vals
    
    
    
    
    
    

def check_unique(df, col, dropna=False):
    
    """Takes in a Pandas DataFrame and specific column name and returns a Pandas DataFrame 
    displaying the unique values in that column as well as the count of each unique value. 
    Default is to also provide a count of NaN values.
    """
    
    import pandas as pd
    
    if dropna:
        unique_vals = pd.DataFrame(df[col].value_counts())
    else:
        unique_vals = pd.DataFrame(df[col].value_counts(dropna=False))
    
    return unique_vals









def check_col_distr(df, col):
    
    """Takes in a Pandas DataFrame and specific column name and returns a Pandas DataFrame 
    displaying the unique values in that column as well as the count of each unique value. 
    Also displays a histogram (Seaborn distplot) showing the distribution of the column values.
    """
    
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    ## check counts of unique values in col
    display(check_unique(df, col))

    ## plot distribution of col
    plt.figure(figsize=(7,5))
    fig = sns.distplot(df[col])
    
    return fig








def plot_box(feature, data, target='seasonal_vaccine'):
    
    """ Takes in a feature/ column name, the DataFrame containing the column, and the target variable 
    (default for this project is 'seasonal_vaccine') and returns a boxplot for that feature grouped by
    vaccination status.
    """
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(7,5))
    fig = sns.boxplot(x=target, 
                      y=feature, 
                      data=data, 
                      palette='nipy_spectral');
    fig.set_title('Vaccinated vs {}'.format(feature), fontsize=16, weight='bold')
    fig.set_xlabel('Vaccine', fontsize=14, weight='bold')
    fig.set_ylabel(feature, fontsize=14, weight='bold')
    
    return fig









def plot_bar(feature, data, target='seasonal_vaccine', hue='seasonal_vaccine', show_legend=False):
    
    """Takes in a feature/ column name, the DataFrame containing the column, and the target variable 
    (default for this project is 'seasonal_vaccine') and returns a barplot for that feature grouped by
    vaccination status.
    """
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(7,5))
    fig = sns.barplot(x=target,
                      y=feature,
                      palette='nipy_spectral',
                      hue=hue,
                      data=data)
    fig.set_title('Vaccinated vs {}'.format(feature), fontsize=16, weight='bold')
    fig.set_xlabel('Vaccine', fontsize=14, weight='bold')
    fig.set_ylabel(feature, fontsize=14, weight='bold')
    
    if show_legend==False:
        fig.get_legend().remove()
    
    return fig








def plot_reg(feature, data, category=None, target='seasonal_vaccine'):
    
    """Takes in a feature/ column name, the DataFrame containing the column, and the target variable 
    (default for this project is 'seasonal_vaccine') and returns a regplot for that feature plotted against 
    vaccination status. Can provide a categorical variable column to plot multiple regression lines of varying
    color for each category in that column.
    """
    
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
    
    """Takes in a feature/ column name, the DataFrame containing the column, and the target variable 
    (default for this project is 'seasonal_vaccine') and returns both a boxplot and barplot for that 
    feature grouped by vaccination status.
    """
    
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
    
    return fig, (ax1,ax2)
    
    
    
    
    
    
def eval_classifier(clf, X_test, y_test, model_descr='',
                    target_labels=['No Vacc', 'Vaccine'],
                    cmap='Blues', normalize='true', save=False, fig_name=None):
    
    """Given an sklearn classification model (already fit to training data), test features, and test labels,
       displays sklearn.metrics classification report, confusion matrix, and ROC curve. A description of the model 
       can be provided to model_descr to customize the title of the classification report.
    """
    
    from sklearn.metrics import classification_report, plot_confusion_matrix, plot_roc_curve
    
    
    folder = '/Users/maxsteele/FlatIron-DS-CourseMaterials/Mod3/Mod3_Project/recloned/dsc-mod-3-project-v2-1-onl01-dtsc-ft-070620'
    fig_filepath = folder+'/Figures/'
    
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
    
    if save:
        plt.savefig(fig_filepath+fig_name)
    
    fig.tight_layout()
    plt.show()

    return fig, axes
    
    
    
    
    
    
def fit_grid_clf(model, params, X_train, y_train, score='accuracy'):
    
    """Given an sklearn classification model, hyperparameter grid, X and y training data, 
       and a GridSearchCV scoring metric (default is 'accuracy', which is the default metric for 
       GridSearchCV), fits a grid search of the specified parameters on the training data and 
       returns the grid object.
    """
    
    from sklearn.model_selection import GridSearchCV
    
    grid = GridSearchCV(model, params, scoring=score, cv=3, n_jobs=-1)
    grid.fit(X_train, y_train)
    
    return grid
    
    
    
    
    
    
    
    
def plot_logreg_coeffs(model, feature_names, model_step='logreg',
                       title='Logistic Regression Coefficients',
                       save=False, fig_name=None):

    """Given an sklearn Logistic Regression Classifier already fit to training data as well as a list of 
       feature names for the model. Returns a figure with two subplots. 
       The first plot displays the top 20 largest (generally positive coefficients)
       and the second displays the last 20 (generally the most influential negative coefficients).
    """
    
    import pandas as pd
    from sklearn.model_selection import GridSearchCV
    import matplotlib.pyplot as plt
    
    folder = '/Users/maxsteele/FlatIron-DS-CourseMaterials/Mod3/Mod3_Project/recloned/dsc-mod-3-project-v2-1-onl01-dtsc-ft-070620'
    fig_filepath = folder+'/Figures/'
    
    logreg_coeffs = model.named_steps[model_step].coef_
    sorted_idx = logreg_coeffs.argsort()

    importance = pd.Series(logreg_coeffs[0], index=feature_names)
    fig, axes = plt.subplots(2, figsize=(12,10))
    importance.sort_values().tail(20).plot(kind='barh', ax=axes[0])
    importance.sort_values().head(20).plot(kind='barh', ax=axes[1])
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    if save:
        plt.savefig(fig_filepath+fig_name)
    
    return fig, axes
    
    
    
    
    
    
def plot_feat_importance(clf, model_step_name, feature_names, model_title='', save=False, fig_name=None):
    
    """Takes in an sklearn classifier already fit to training data, the name of the step for that model
       in the modeling pipeline, the feature names, and optionally a title describing the model. 
       Returns a horizontal barplot showing the top 20 most important features in descending order.
    """

    import pandas as pd
    from sklearn.model_selection import GridSearchCV
    import matplotlib.pyplot as plt
    
    folder = '/Users/maxsteele/FlatIron-DS-CourseMaterials/Mod3/Mod3_Project/recloned/dsc-mod-3-project-v2-1-onl01-dtsc-ft-070620'
    fig_filepath = folder+'/Figures/'
    
    feature_importances = (
        clf.named_steps[model_step_name].feature_importances_)

    sorted_idx = feature_importances.argsort()
    
    importance = pd.Series(feature_importances, index=feature_names)
    plt.figure(figsize=(12,10))
    fig = importance.sort_values().tail(20).plot(kind='barh')
    fig.set_title('{} Feature Importances'.format(model_title), fontsize=18, fontweight='bold')
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12)
    
    if save:
        plt.savefig(fig_filepath+fig_name)

    plt.show()
    
    return fig

    
    
    
    
    
    
def plot_count_by_grp(group, data, hue='seasonal_vaccine',
                      labels=['No Vacc', 'Vaccine'], title='',
                      y_label='# of Respondents', x_label='',
                      x_tick_labels=False, rotate=True,
                      grp_order=None):
    
    """Takes in the name of a column to group by and the DataFrame, and returns a countplot with
       bars color-coded by target variable (for this project, seasonal vaccination status, with corresponding
       default labels 'No Vacc' and 'Vaccine'). Provides options for changing group names on the x-tick
       labels, changing the group order, and rotating the x-tick labels.
    """
    
    import pandas as pd
    from sklearn.model_selection import GridSearchCV
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    font_dict = {}
    font_dict['title'] = {'fontsize':18, 'fontweight':'bold'}
    font_dict['axis_label'] = {'fontsize':14, 'fontweight':'bold'}
    font_dict['ticks'] = {'size':14}
    font_dict['legend'] = {'fontsize':12}
    
    plt.figure(figsize=(8,6))
    fig = sns.countplot(x=group, hue=hue,
                  data=data, palette='nipy_spectral',
                      order=grp_order)
    fig.set_title('Vaccination By {}'.format(title), fontdict=font_dict['title'])
    fig.set_xlabel(x_label, fontdict=font_dict['axis_label'])
    fig.set_ylabel(y_label, fontdict=font_dict['axis_label'])
    fig.tick_params(labelsize=font_dict['ticks']['size'])
    
    if rotate:
        fig.set_xticklabels(fig.get_xticklabels(), rotation=45)
    if x_tick_labels:
        fig.set_xticklabels(x_tick_labels)

    fig.legend(labels=labels, fontsize=font_dict['legend']['fontsize'])
    plt.show();
    
    return fig







def plot_final_1(x, df, group_order=None, x_label='',
               title='', labels=['No Vacc', 'Vaccine'],
               x_tick_labels=False, target='seasonal_vaccine',
               figsize=(6,5), save=False, fig_name=None):
    
    folder = '/Users/maxsteele/FlatIron-DS-CourseMaterials/Mod3/Mod3_Project/recloned/dsc-mod-3-project-v2-1-onl01-dtsc-ft-070620'
    fig_filepath = folder+'/Figures/'
    
    palette='nipy_spectral'
    font_dict = {}
    font_dict['title'] = {'fontsize':18, 'fontweight':'bold'}
    font_dict['axis_label'] = {'fontsize':14, 'fontweight':'bold'}
    font_dict['ticks'] = {'size':14}
    font_dict['legend'] = {'fontsize':12}

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    sns.countplot(x=x, hue=target, data=df, palette=palette,
                      order=group_order, ax=ax)
    ax.set_title('Vaccination By {}'.format(title), fontdict=font_dict['title'])
    ax.set_xlabel(x_label, fontdict=font_dict['axis_label'])
    ax.set_ylabel('# of Respondents', fontdict=font_dict['axis_label'])
    ax.tick_params(labelsize=font_dict['ticks']['size'])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.legend(labels=labels, fontsize=font_dict['legend']['fontsize'])
    if x_tick_labels:
        ax.set_xticklabels(x_tick_labels)

    if save:
        plt.savefig(fig_filepath+fig_name)

    plt.show();
    
    return fig, ax 








def plot_final_2(x1, x2, df1, df2, group1_order=None, group2_order=None,
               title1='', title2='', x1_label='', x2_label='',
               labels=['No Vacc', 'Vaccine'],
               x1_tick_labels=False, x2_tick_labels=False,
               target='seasonal_vaccine', figsize=(12,6), 
               save=False, fig_name=None):
    
    folder = '/Users/maxsteele/FlatIron-DS-CourseMaterials/Mod3/Mod3_Project/recloned/dsc-mod-3-project-v2-1-onl01-dtsc-ft-070620'
    fig_filepath = folder+'/Figures/'
    
    palette='nipy_spectral'
    font_dict = {}
    font_dict['title'] = {'fontsize':18, 'fontweight':'bold'}
    font_dict['axis_label'] = {'fontsize':14, 'fontweight':'bold'}
    font_dict['ticks'] = {'size':14}
    font_dict['legend'] = {'fontsize':12}

    fig,(ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    sns.countplot(x=x1, hue=target, data=df1, palette=palette,
                      order=group1_order, ax=ax1)
    ax1.set_title('Vaccination By {}'.format(title1), fontdict=font_dict['title'])
    ax1.set_xlabel(x1_label, fontdict=font_dict['axis_label'])
    ax1.set_ylabel('# of Respondents', fontdict=font_dict['axis_label'])
    ax1.tick_params(labelsize=font_dict['ticks']['size'])
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
    ax1.legend_.remove()
    if x1_tick_labels:
        ax1.set_xticklabels(x1_tick_labels)

        
        
    sns.countplot(x=x2, hue=target, data=df2, palette=palette,
                      order=group2_order, ax=ax2);
    ax2.set_title('Vaccination By {}'.format(title2), fontdict=font_dict['title'])
    ax2.set_xlabel(x2_label, fontdict=font_dict['axis_label'])
    ax2.set_ylabel('', fontdict=font_dict['axis_label'])
    ax2.tick_params(labelsize=font_dict['ticks']['size'])
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
    ax2.legend(labels=labels, fontsize=font_dict['legend']['fontsize'])
    if x2_tick_labels:
        ax2.set_xticklabels(x2_tick_labels)

    if save:
        plt.savefig(fig_filepath+fig_name)
        
    plt.tight_layout()
    plt.show();
    
    return fig,(ax1,ax2) 