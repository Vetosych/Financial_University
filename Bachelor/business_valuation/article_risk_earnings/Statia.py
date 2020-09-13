#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')
from IPython.display import SVG
from graphviz import Source
from IPython.display import display
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from IPython.display import HTML
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
style = "<style>svg{width:50% !important;height:50% !important;}</style>"
HTML(style)


# In[2]:


from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png', 'pdf')


# In[3]:


statia = pd.read_excel('C:\\Users\\Виталий\\Desktop\\Питон\\Статья_ОБ\\data_ex.xlsx')


# In[4]:


statia.head()


# In[5]:


statia.dtypes


# In[16]:


X_1 = statia[['Beta','MARKET CAP REAL','AVG VOLUME REAL']]
y_1 = statia['PERF YEAR CLASS']
clf_1 = tree.DecisionTreeClassifier(criterion='entropy',max_depth=2)


# In[17]:


clf_1.fit(X_1, y_1)


# In[21]:


clf_1.score(X_1, y_1)


# In[18]:


graph = Source(tree.export_graphviz(clf_1, out_file=None,
                                   feature_names=list(X_1),
                                   class_names=['VeryBad','Bad','Avg','Good','VeryGood'],
                                   filled = True))
display(SVG(graph.pipe(format='svg')))


# In[22]:


X_2 = statia[['Beta','MARKET CAP REAL','AVG VOLUME REAL']]
y_2 = statia['PERF YEAR CLASS']
clf_2 = tree.DecisionTreeClassifier(criterion='entropy')


# In[24]:


clf_2.fit(X_2, y_2)


# In[25]:


clf_2.score(X_2, y_2)


# In[26]:


graph = Source(tree.export_graphviz(clf_2, out_file=None,
                                   feature_names=list(X_2),
                                   class_names=['VeryBad','Bad','Avg','Good','VeryGood'],
                                   filled = True))
display(SVG(graph.pipe(format='svg')))


# In[87]:


X_3 = statia[['Beta','MARKET CAP REAL','AVG VOLUME REAL']]
y_3 = statia['PERF YEAR CLASS']
clf_3 = tree.DecisionTreeClassifier(criterion='entropy',
                                    max_depth=3)


# In[88]:


clf_3.fit(X_3, y_3)


# In[89]:


clf_3.score(X_3, y_3)


# In[90]:


graph = Source(tree.export_graphviz(clf_3, out_file=None,
                                   feature_names=list(X_3),
                                   class_names=['VeryBad','Bad','Avg','Good','VeryGood'],
                                   filled = True))
display(SVG(graph.pipe(format='svg')))


# In[91]:


X3_train, X3_test, y3_train, y3_test = train_test_split(X_3, y_3)


# In[92]:


parametrs = {'max_depth':range(2,5)}


# In[93]:


best_clf_3 = tree.DecisionTreeClassifier(criterion='entropy')
best_clf_3


# In[94]:


grid_search_cv_clf_3 = GridSearchCV(best_clf_3, parametrs, cv=5)
grid_search_cv_clf_3.fit(X3_train, y3_train)


# In[95]:


grid_search_cv_clf_3.best_params_


# In[96]:


the_best_clf_3 = grid_search_cv_clf_3.best_estimator_


# In[97]:


the_best_clf_3.score(X3_test, y3_test)


# In[101]:


graph = Source(tree.export_graphviz(the_best_clf_3, out_file=None,
                                   feature_names=list(X3_test),
                                   class_names=['VeryBad','Bad','Avg','Good','VeryGood'],
                                   filled = True))
display(SVG(graph.pipe(format='svg')))


# In[103]:


X_4 = statia[['Beta','MARKET CAP REAL']]
y_4 = statia['PERF YEAR CLASS']
X4_train, X4_test, y4_train, y4_test = train_test_split(X_4, y_4)
parametrs = {'max_depth':range(2,5)}
best_clf_4 = tree.DecisionTreeClassifier(criterion='entropy')
grid_search_cv_clf_4 = GridSearchCV(best_clf_4, parametrs, cv=5)
grid_search_cv_clf_4.fit(X4_train, y4_train)
the_best_clf_4 = grid_search_cv_clf_4.best_estimator_
the_best_clf_4.score(X4_test, y4_test)


# In[104]:


graph = Source(tree.export_graphviz(the_best_clf_4, out_file=None,
                                   feature_names=list(X4_test),
                                   class_names=['VeryBad','Bad','Avg','Good','VeryGood'],
                                   filled = True))
display(SVG(graph.pipe(format='svg')))


# In[105]:


X_5 = statia[['Beta','AVG VOLUME REAL']]
y_5 = statia['PERF YEAR CLASS']
X5_train, X5_test, y5_train, y5_test = train_test_split(X_5, y_5)
parametrs = {'max_depth':range(2,5)}
best_clf_5 = tree.DecisionTreeClassifier(criterion='entropy')
grid_search_cv_clf_5 = GridSearchCV(best_clf_5, parametrs, cv=5)
grid_search_cv_clf_5.fit(X5_train, y5_train)
the_best_clf_5 = grid_search_cv_clf_5.best_estimator_
the_best_clf_5.score(X5_test, y5_test)


# In[106]:


graph = Source(tree.export_graphviz(the_best_clf_5, out_file=None,
                                   feature_names=list(X5_test),
                                   class_names=['VeryBad','Bad','Avg','Good','VeryGood'],
                                   filled = True))
display(SVG(graph.pipe(format='svg')))


# In[107]:


X_6 = statia[['MARKET CAP REAL','AVG VOLUME REAL']]
y_6 = statia['PERF YEAR CLASS']
X6_train, X6_test, y6_train, y6_test = train_test_split(X_6, y_6)
parametrs = {'max_depth':range(2,5)}
best_clf_6 = tree.DecisionTreeClassifier(criterion='entropy')
grid_search_cv_clf_6 = GridSearchCV(best_clf_6, parametrs, cv=5)
grid_search_cv_clf_6.fit(X6_train, y6_train)
the_best_clf_6 = grid_search_cv_clf_6.best_estimator_
the_best_clf_6.score(X6_test, y6_test)


# In[108]:


graph = Source(tree.export_graphviz(the_best_clf_6, out_file=None,
                                   feature_names=list(X6_test),
                                   class_names=['VeryBad','Bad','Avg','Good','VeryGood'],
                                   filled = True))
display(SVG(graph.pipe(format='svg')))


# In[109]:


statia_USA = statia[(statia['REGION'] == 'USA')]


# In[111]:


statia_USA.head()


# In[112]:


statia_NON_USA = statia[(statia['REGION'] != 'USA')]


# In[113]:


statia_NON_USA.head()


# In[116]:


statia_basic_materials = statia[(statia['Sector'] == 'Basic Materials')]


# In[117]:


statia_basic_materials.head()


# In[118]:


statia_Conglomerates = statia[(statia['Sector'] == 'Conglomerates')]
statia_Conglomerates.head()


# In[119]:


statia_Consumer_Goods = statia[(statia['Sector'] == 'Consumer Goods')]
statia_Consumer_Goods.head()


# In[120]:


statia_Financial = statia[(statia['Sector'] == 'Financial')]
statia_Financial.head()


# In[121]:


statia_Healthcare = statia[(statia['Sector'] == 'Healthcare')]
statia_Healthcare.head()


# In[123]:


statia_Industrial_Goods = statia[(statia['Sector'] == 'Industrial Goods')]
statia_Industrial_Goods.head()


# In[124]:


statia_Services = statia[(statia['Sector'] == 'Services')]
statia_Services.head()


# In[125]:


statia_Technology = statia[(statia['Sector'] == 'Technology')]
statia_Technology.head()


# In[126]:


statia_Utilities = statia[(statia['Sector'] == 'Utilities')]
statia_Utilities.head()


# In[127]:


statia_cap_big = statia[(statia['MARKET CAP REAL'] >= 1070000000)]
statia_cap_big.head()


# In[128]:


statia_cap_sml = statia[(statia['MARKET CAP REAL'] < 1070000000)]
statia_cap_sml.head()


# In[129]:


statia_vol_big = statia[(statia['AVG VOLUME REAL'] >= 334390)]
statia_vol_big.head()


# In[130]:


statia_vol_sml = statia[(statia['AVG VOLUME REAL'] < 334390)]
statia_vol_sml.head()


# In[131]:


X_7 = statia_USA[['Beta','MARKET CAP REAL','AVG VOLUME REAL']]
y_7 = statia_USA['PERF YEAR CLASS']
X7_train, X7_test, y7_train, y7_test = train_test_split(X_7, y_7)
parametrs = {'max_depth':range(2,5)}
best_clf_7 = tree.DecisionTreeClassifier(criterion='entropy')
grid_search_cv_clf_7 = GridSearchCV(best_clf_7, parametrs, cv=5)
grid_search_cv_clf_7.fit(X7_train, y7_train)
the_best_clf_7 = grid_search_cv_clf_7.best_estimator_
the_best_clf_7.score(X7_test, y7_test)


# In[132]:


graph = Source(tree.export_graphviz(the_best_clf_7, out_file=None,
                                   feature_names=list(X7_test),
                                   class_names=['VeryBad','Bad','Avg','Good','VeryGood'],
                                   filled = True))
display(SVG(graph.pipe(format='svg')))


# In[133]:


X_8 = statia_NON_USA[['Beta','MARKET CAP REAL','AVG VOLUME REAL']]
y_8 = statia_NON_USA['PERF YEAR CLASS']
X8_train, X8_test, y8_train, y8_test = train_test_split(X_8, y_8)
parametrs = {'max_depth':range(2,5)}
best_clf_8 = tree.DecisionTreeClassifier(criterion='entropy')
grid_search_cv_clf_8 = GridSearchCV(best_clf_8, parametrs, cv=5)
grid_search_cv_clf_8.fit(X8_train, y8_train)
the_best_clf_8 = grid_search_cv_clf_8.best_estimator_
the_best_clf_8.score(X8_test, y8_test)


# In[134]:


graph = Source(tree.export_graphviz(the_best_clf_8, out_file=None,
                                   feature_names=list(X8_test),
                                   class_names=['VeryBad','Bad','Avg','Good','VeryGood'],
                                   filled = True))
display(SVG(graph.pipe(format='svg')))


# In[135]:


X_9 = statia_basic_materials[['Beta','MARKET CAP REAL','AVG VOLUME REAL']]
y_9 = statia_basic_materials['PERF YEAR CLASS']
X9_train, X9_test, y9_train, y9_test = train_test_split(X_9, y_9)
parametrs = {'max_depth':range(2,5)}
best_clf_9 = tree.DecisionTreeClassifier(criterion='entropy')
grid_search_cv_clf_9 = GridSearchCV(best_clf_9, parametrs, cv=5)
grid_search_cv_clf_9.fit(X9_train, y9_train)
the_best_clf_9 = grid_search_cv_clf_9.best_estimator_
the_best_clf_9.score(X9_test, y9_test)


# In[142]:


graph = Source(tree.export_graphviz(the_best_clf_9, out_file=None,
                                   feature_names=list(X9_test),
                                   class_names=['VeryBad','Bad','Avg','Good','VeryGood'],
                                   filled = True))
display(SVG(graph.pipe(format='svg')))


# In[141]:


X_10 = statia_Conglomerates[['Beta','MARKET CAP REAL','AVG VOLUME REAL']]
y_10 = statia_Conglomerates['PERF YEAR CLASS']
X10_train, X10_test, y10_train, y10_test = train_test_split(X_10, y_10)
parametrs = {'max_depth':range(2,5)}
best_clf_10 = tree.DecisionTreeClassifier(criterion='entropy')
grid_search_cv_clf_10 = GridSearchCV(best_clf_10, parametrs, cv=2)
grid_search_cv_clf_10.fit(X10_train, y10_train)
the_best_clf_10 = grid_search_cv_clf_10.best_estimator_
the_best_clf_10.score(X10_test, y10_test)


# In[143]:


graph = Source(tree.export_graphviz(the_best_clf_10, out_file=None,
                                   feature_names=list(X10_test),
                                   class_names=['VeryBad','Bad','Avg','Good','VeryGood'],
                                   filled = True))
display(SVG(graph.pipe(format='svg')))


# In[144]:


X_11 = statia_Consumer_Goods[['Beta','MARKET CAP REAL','AVG VOLUME REAL']]
y_11 = statia_Consumer_Goods['PERF YEAR CLASS']
X11_train, X11_test, y11_train, y11_test = train_test_split(X_11, y_11)
parametrs = {'max_depth':range(2,5)}
best_clf_11 = tree.DecisionTreeClassifier(criterion='entropy')
grid_search_cv_clf_11 = GridSearchCV(best_clf_11, parametrs, cv=5)
grid_search_cv_clf_11.fit(X11_train, y11_train)
the_best_clf_11 = grid_search_cv_clf_11.best_estimator_
the_best_clf_11.score(X11_test, y11_test)


# In[145]:


graph = Source(tree.export_graphviz(the_best_clf_11, out_file=None,
                                   feature_names=list(X11_test),
                                   class_names=['VeryBad','Bad','Avg','Good','VeryGood'],
                                   filled = True))
display(SVG(graph.pipe(format='svg')))


# In[146]:


X_12 = statia_Financial[['Beta','MARKET CAP REAL','AVG VOLUME REAL']]
y_12 = statia_Financial['PERF YEAR CLASS']
X12_train, X12_test, y12_train, y12_test = train_test_split(X_12, y_12)
parametrs = {'max_depth':range(2,5)}
best_clf_12 = tree.DecisionTreeClassifier(criterion='entropy')
grid_search_cv_clf_12 = GridSearchCV(best_clf_12, parametrs, cv=5)
grid_search_cv_clf_12.fit(X12_train, y12_train)
the_best_clf_12 = grid_search_cv_clf_12.best_estimator_
the_best_clf_12.score(X12_test, y12_test)


# In[147]:


graph = Source(tree.export_graphviz(the_best_clf_12, out_file=None,
                                   feature_names=list(X12_test),
                                   class_names=['VeryBad','Bad','Avg','Good','VeryGood'],
                                   filled = True))
display(SVG(graph.pipe(format='svg')))


# In[148]:


X_13 = statia_Healthcare[['Beta','MARKET CAP REAL','AVG VOLUME REAL']]
y_13 = statia_Healthcare['PERF YEAR CLASS']
X13_train, X13_test, y13_train, y13_test = train_test_split(X_13, y_13)
parametrs = {'max_depth':range(2,5)}
best_clf_13 = tree.DecisionTreeClassifier(criterion='entropy')
grid_search_cv_clf_13 = GridSearchCV(best_clf_13, parametrs, cv=5)
grid_search_cv_clf_13.fit(X13_train, y13_train)
the_best_clf_13 = grid_search_cv_clf_13.best_estimator_
the_best_clf_13.score(X13_test, y13_test)


# In[149]:


graph = Source(tree.export_graphviz(the_best_clf_13, out_file=None,
                                   feature_names=list(X13_test),
                                   class_names=['VeryBad','Bad','Avg','Good','VeryGood'],
                                   filled = True))
display(SVG(graph.pipe(format='svg')))


# In[150]:


X_14 = statia_Industrial_Goods[['Beta','MARKET CAP REAL','AVG VOLUME REAL']]
y_14 = statia_Industrial_Goods['PERF YEAR CLASS']
X14_train, X14_test, y14_train, y14_test = train_test_split(X_14, y_14)
parametrs = {'max_depth':range(2,5)}
best_clf_14 = tree.DecisionTreeClassifier(criterion='entropy')
grid_search_cv_clf_14 = GridSearchCV(best_clf_14, parametrs, cv=5)
grid_search_cv_clf_14.fit(X14_train, y14_train)
the_best_clf_14 = grid_search_cv_clf_14.best_estimator_
the_best_clf_14.score(X14_test, y14_test)


# In[151]:


graph = Source(tree.export_graphviz(the_best_clf_14, out_file=None,
                                   feature_names=list(X14_test),
                                   class_names=['VeryBad','Bad','Avg','Good','VeryGood'],
                                   filled = True))
display(SVG(graph.pipe(format='svg')))


# In[153]:


X_15 = statia_Services[['Beta','MARKET CAP REAL','AVG VOLUME REAL']]
y_15 = statia_Services['PERF YEAR CLASS']
X15_train, X15_test, y15_train, y15_test = train_test_split(X_15, y_15)
parametrs = {'max_depth':range(2,5)}
best_clf_15 = tree.DecisionTreeClassifier(criterion='entropy')
grid_search_cv_clf_15 = GridSearchCV(best_clf_15, parametrs, cv=5)
grid_search_cv_clf_15.fit(X15_train, y15_train)
the_best_clf_15 = grid_search_cv_clf_15.best_estimator_
the_best_clf_15.score(X15_test, y15_test)


# In[154]:


graph = Source(tree.export_graphviz(the_best_clf_15, out_file=None,
                                   feature_names=list(X15_test),
                                   class_names=['VeryBad','Bad','Avg','Good','VeryGood'],
                                   filled = True))
display(SVG(graph.pipe(format='svg')))


# In[155]:


X_16 = statia_Technology[['Beta','MARKET CAP REAL','AVG VOLUME REAL']]
y_16 = statia_Technology['PERF YEAR CLASS']
X16_train, X16_test, y16_train, y16_test = train_test_split(X_16, y_16)
parametrs = {'max_depth':range(2,5)}
best_clf_16 = tree.DecisionTreeClassifier(criterion='entropy')
grid_search_cv_clf_16 = GridSearchCV(best_clf_16, parametrs, cv=5)
grid_search_cv_clf_16.fit(X16_train, y16_train)
the_best_clf_16 = grid_search_cv_clf_16.best_estimator_
the_best_clf_16.score(X16_test, y16_test)


# In[156]:


graph = Source(tree.export_graphviz(the_best_clf_16, out_file=None,
                                   feature_names=list(X16_test),
                                   class_names=['VeryBad','Bad','Avg','Good','VeryGood'],
                                   filled = True))
display(SVG(graph.pipe(format='svg')))


# In[157]:


X_17 = statia_Utilities[['Beta','MARKET CAP REAL','AVG VOLUME REAL']]
y_17 = statia_Utilities['PERF YEAR CLASS']
X17_train, X17_test, y17_train, y17_test = train_test_split(X_17, y_17)
parametrs = {'max_depth':range(2,5)}
best_clf_17 = tree.DecisionTreeClassifier(criterion='entropy')
grid_search_cv_clf_17 = GridSearchCV(best_clf_17, parametrs, cv=5)
grid_search_cv_clf_17.fit(X17_train, y17_train)
the_best_clf_17 = grid_search_cv_clf_17.best_estimator_
the_best_clf_17.score(X17_test, y17_test)


# In[158]:


graph = Source(tree.export_graphviz(the_best_clf_17, out_file=None,
                                   feature_names=list(X17_test),
                                   class_names=['VeryBad','Bad','Avg','Good','VeryGood'],
                                   filled = True))
display(SVG(graph.pipe(format='svg')))


# In[159]:


X_18 = statia_cap_big[['Beta','MARKET CAP REAL','AVG VOLUME REAL']]
y_18 = statia_cap_big['PERF YEAR CLASS']
X18_train, X18_test, y18_train, y18_test = train_test_split(X_18, y_18)
parametrs = {'max_depth':range(2,5)}
best_clf_18 = tree.DecisionTreeClassifier(criterion='entropy')
grid_search_cv_clf_18 = GridSearchCV(best_clf_18, parametrs, cv=5)
grid_search_cv_clf_18.fit(X18_train, y18_train)
the_best_clf_18 = grid_search_cv_clf_18.best_estimator_
the_best_clf_18.score(X18_test, y18_test)


# In[160]:


graph = Source(tree.export_graphviz(the_best_clf_18, out_file=None,
                                   feature_names=list(X18_test),
                                   class_names=['VeryBad','Bad','Avg','Good','VeryGood'],
                                   filled = True))
display(SVG(graph.pipe(format='svg')))


# In[161]:


X_19 = statia_cap_sml[['Beta','MARKET CAP REAL','AVG VOLUME REAL']]
y_19 = statia_cap_sml['PERF YEAR CLASS']
X19_train, X19_test, y19_train, y19_test = train_test_split(X_19, y_19)
parametrs = {'max_depth':range(2,5)}
best_clf_19 = tree.DecisionTreeClassifier(criterion='entropy')
grid_search_cv_clf_19 = GridSearchCV(best_clf_19, parametrs, cv=5)
grid_search_cv_clf_19.fit(X19_train, y19_train)
the_best_clf_19 = grid_search_cv_clf_19.best_estimator_
the_best_clf_19.score(X19_test, y19_test)


# In[162]:


graph = Source(tree.export_graphviz(the_best_clf_19, out_file=None,
                                   feature_names=list(X19_test),
                                   class_names=['VeryBad','Bad','Avg','Good','VeryGood'],
                                   filled = True))
display(SVG(graph.pipe(format='svg')))


# In[163]:


X_20 = statia_vol_big[['Beta','MARKET CAP REAL','AVG VOLUME REAL']]
y_20 = statia_vol_big['PERF YEAR CLASS']
X20_train, X20_test, y20_train, y20_test = train_test_split(X_20, y_20)
parametrs = {'max_depth':range(2,5)}
best_clf_20 = tree.DecisionTreeClassifier(criterion='entropy')
grid_search_cv_clf_20 = GridSearchCV(best_clf_20, parametrs, cv=5)
grid_search_cv_clf_20.fit(X20_train, y20_train)
the_best_clf_20 = grid_search_cv_clf_20.best_estimator_
the_best_clf_20.score(X20_test, y20_test)


# In[164]:


graph = Source(tree.export_graphviz(the_best_clf_20, out_file=None,
                                   feature_names=list(X20_test),
                                   class_names=['VeryBad','Bad','Avg','Good','VeryGood'],
                                   filled = True))
display(SVG(graph.pipe(format='svg')))


# In[165]:


X_21 = statia_vol_sml[['Beta','MARKET CAP REAL','AVG VOLUME REAL']]
y_21 = statia_vol_sml['PERF YEAR CLASS']
X21_train, X21_test, y21_train, y21_test = train_test_split(X_21, y_21)
parametrs = {'max_depth':range(2,5)}
best_clf_21 = tree.DecisionTreeClassifier(criterion='entropy')
grid_search_cv_clf_21 = GridSearchCV(best_clf_21, parametrs, cv=5)
grid_search_cv_clf_21.fit(X21_train, y21_train)
the_best_clf_21 = grid_search_cv_clf_21.best_estimator_
the_best_clf_21.score(X21_test, y21_test)


# In[166]:


graph = Source(tree.export_graphviz(the_best_clf_21, out_file=None,
                                   feature_names=list(X21_test),
                                   class_names=['VeryBad','Bad','Avg','Good','VeryGood'],
                                   filled = True))
display(SVG(graph.pipe(format='svg')))


# In[ ]:




