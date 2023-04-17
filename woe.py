import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
 
    

def woe_discrete(X_var, y, sort = True, plot = True, rot_ax = 30, fontsize_ax = 12):
    
    """ 
        Calculate WOE - work in progress
        
        TODO: 
            Exceptions 
            Documentation
            Add to class?
            
    """
    
    df = pd.concat([X_var,y], axis = 1)

    df = df.groupby(X_var.name).agg(['count', 'mean'])
    df.columns = ['n_obs','prop_good']
    df = df.reset_index()
    df['n_good'] = df['prop_good'] * df['n_obs']
    df['n_bad'] = (1 - df['prop_good']) * df['n_obs']
    df['prop_n_good'] = df['n_good'] / df['n_good'].sum()
    df['prop_n_bad'] = df['n_bad'] / df['n_bad'].sum()
    df['WoE'] = np.log(df['prop_n_good'] / df['prop_n_bad'])
    
    if sort:
        df = df.sort_values(['WoE'])
    
    df = df.reset_index(drop = True)

    IV = sum((df['prop_n_good'] - df['prop_n_bad']) * df['WoE'])
    
    if plot: 
        plt.figure(figsize=(28, 6))       
          
        p1 = plt.plot(df[X_var.name], df['WoE'], marker = 'o', linestyle = '--', color = 'k')
        plt.xlabel(X_var.name)
        plt.ylabel('Weight of Evidence')
        plt.title(f'Weight of Evidence by {X_var.name}, IV = {round(IV,2)}')
        plt.xticks(rotation = rot_ax, fontsize = fontsize_ax)
              
        axes2 = plt.twinx()
        p2 = plt.bar(df[X_var.name], df['n_good'], alpha=0.3)
        p3 = plt.bar(df[X_var.name], df['n_bad'], alpha=0.3)
        axes2.set_ylabel('Count (non)fraud [log_scale]')
        axes2.set_yscale("log")
#         plt.legend(bbox_to_anchor=(1.1, 1.05))  
        plt.legend([p2,p3], ['Non-fraud transaction [log]','Fraud transaction [log]'] , loc =2 )
    
    return df
