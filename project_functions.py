import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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



def plot_box(feature, target='seasonal_vaccine', data=df):
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



def plot_bar(feature, target='seasonal_vaccine', hue='seasonal_vaccine', data=df, show_legend=False):
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



def plot_reg(feature, category=None, target='seasonal_vaccine', data=df):
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


def plot_bb(feature, target='seasonal_vaccine', data=df):
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