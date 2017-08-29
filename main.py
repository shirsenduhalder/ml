import os 
import pandas as pd 

HOUSING_PATH = 'datasets/housing/'

def load_housing_data(housing_path = HOUSING_PATH):
	csv_path = os.path.join(housing_path,'housing.csv')
	return pd.read_csv(csv_path)

housing_initial=load_housing_data()
# print(housing_initial.head())
# print(housing_initial['ocean_proximity'].value_counts())

import matplotlib.pyplot as plt 
# housing_initial.hist(bins=50,figsize=(20,15))
# plt.show()

import numpy as np

# def create_test_set(data,test_size=0.2):
# 	shuffled_indices = np.random.permutation(len(data))
# 	print(shuffled_indices)
# 	test_set_size = int(len(data)*test_size)
# 	test_indices = shuffled_indices[:test_set_size]
# 	train_indices = shuffled_indices[test_set_size:]
# 	return data.iloc[train_indices], data.iloc[test_indices]

from sklearn.model_selection import train_test_split
# housing_initial = housing_initial.reset_index()
housing_initial['id'] = housing_initial["longitude"]*1000 + housing_initial["latitude"]
housing_initial['income_cat'] = np.ceil(housing_initial["median_income"]/1.5)
housing_initial['income_cat'].where(housing_initial["income_cat"]<5,5.0,inplace=True)
train_set,test_set = train_test_split(housing_initial,test_size=0.2,random_state=42)
# print(housing_initial['income_cat'].value_counts()/len(housing_initial))
# print(train_set['income_cat'].value_counts()/len(housing_initial))

from sklearn.model_selection import StratifiedShuffleSplit

split_data = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

for train_index,test_index in split_data.split(housing_initial,housing_initial['income_cat']):
	strat_train_set = housing_initial.loc[train_index]
	strat_test_set = housing_initial.loc[test_index]

# print(strat_train_set['income_cat'].value_counts()/len(housing))

# print(len(test_set),'\t',len(train_set))
# print(test_set.head())

for set in (strat_train_set,strat_test_set):
	set.drop(['income_cat'],axis=1,inplace=True)

housing = strat_train_set.copy()
housing.plot(kind="scatter",x="longitude",y="latitude",alpha=0.4,
	s=housing["population"]/100,label="population",
	c="median_house_value",cmap=plt.get_cmap('jet'),colorbar=True)
# plt.legend()
# plt.show()

corr_matrix = housing.corr()
# print((corr_matrix))
# print(corr_matrix["median_house_value"].sort_values(ascending=False))

housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]
# print(housing.head())
housing = strat_train_set.drop("median_house_value",1)
# print(housing.head())

housing_labels = strat_train_set["median_house_value"]
# print(housing["ocean_proximity"].value_counts())
bedroom_median = housing['total_bedrooms'].median()
housing["total_bedrooms"].fillna(bedroom_median)

from sklearn.preprocessing import Imputer

# imputer = Imputer(strategy="median")

housing_num = housing.drop("ocean_proximity",1).copy()
# imputer.fit(housing_num)
# # print(housing.head())
# # print(imputer.statistics_)
# # print(housing.median().values)

# X = imputer.transform(housing_num)
# housing_num = pd.DataFrame(X,columns = housing_num.columns)
# print(housing.head())

# Transformer 1
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
housing_cat = housing["ocean_proximity"]
# housing_cat_encoded = encoder.fit_transform(housing_cat)
# print(housing_cat_encoded)

# from sklearn.preprocessing import OneHotEncoder

# encoder = OneHotEncoder()
# housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
# print(housing_cat_1hot)

#Using LabelBinarizer

from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
housing_cat_1hot = encoder.fit_transform(housing_cat)
housing_cat_1hot =pd.DataFrame(housing_cat_1hot)
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin 

class DataFrameSelector(BaseEstimator,TransformerMixin):
	def __init__(self,attribute_names):
		self.attribute_names = attribute_names
	def fit(self,X,y=None):
		return self
	def transform(self,X):
		return X[self.attribute_names].values

rooms_ix,bedrooms_ix,population_ix,household_ix = 3,4,5,6

class combinedAttributesAdder(BaseEstimator,TransformerMixin):
	def __init__(self,add_bedrooms_per_room=True):
		self.add_bedrooms_per_room = add_bedrooms_per_room
	def fit(self,X,y=None):
		return self
	def transform(self,X,y=None):
		rooms_per_household = X[:,rooms_ix]/X[:,household_ix]
		population_per_household = X[:,population_ix]/X[:,household_ix]
		if self.add_bedrooms_per_room:
			bedrooms_per_room = X[:,bedrooms_ix]/X[:,rooms_ix]
			return np.c_[X,rooms_per_household,population_per_household,bedrooms_per_room]
		else:
			return np.c_[X,rooms_per_household,population_per_household]


attr_adder = combinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing_num.values)
housing_extra_attribs = pd.DataFrame(housing_extra_attribs, columns=list(housing_num.columns)+["rooms_per_household", "population_per_household"])
# print(housing_extra_attribs.head())
num_pipeline = Pipeline([
	('imputer',Imputer(strategy="median")),
	('attribs_adder',combinedAttributesAdder()),
	('std_scaler',StandardScaler())])

housing_num_tr = num_pipeline.fit_transform(housing_num)
# print(housing_num.head())

num_attribs = list(housing_num)
total_attribs = list(housing_extra_attribs)
# cat_attribs = ["ocean_proximity"]

num_pipeline1 = Pipeline([
	('selector',DataFrameSelector(num_attribs)),
	('imputer',Imputer(strategy="median")),
	('attribs_adder',combinedAttributesAdder()),
	('std_scaler',StandardScaler()),
	])

# cat_pipeline = Pipeline([
# 				('selector',DataFrameSelector(cat_attribs)),
# 				('label_binarizer',LabelBinarizer()),
# 				])

# full_pipeline = FeatureUnion(transformer_list=[
# 	('num_pipeline',num_pipeline),
# 	('cat_pipeline',cat_pipeline),
# 	])

# # print(housing.head())
# housing_prepared = full_pipeline.fit(housing)
# housing_prepared = full_pipeline.transform(housing)

# # print(housing_prepared.shape)

housing_final = housing_extra_attribs.copy()
housing_final = housing_final.fillna(method='ffill').join(housing_cat_1hot)
# housing_cat_final = housing_cat_1hot.copy()
# housing_final = pd.join(housing_num_final,housing_cat_final)
# print(housing_num_final.head())

# print(housing_num_final.head())
# # print(housing_cat_final.head())

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_final,housing_labels)

from sklearn.metrics import mean_squared_error
lr_predictions = lin_reg.predict(housing_final)
lin_mse = mean_squared_error(housing_labels,lr_predictions)
lin_rmse = np.sqrt(lin_mse)
# print("Linear Regression \n")
# print(lin_rmse)

from sklearn.model_selection import cross_val_score

#cross validation for linear regression

scores = cross_val_score(lin_reg,housing_final,housing_labels,
	scoring="neg_mean_squared_error",cv=10)
lr_rmse_scores = np.sqrt(-scores)

#using decision trees

from sklearn.tree import DecisionTreeRegressor
def display_scores(scores):
	print("Scores: ",scores)
	print("Mean: ",scores.mean())
	print("Standard Deviation: ",scores.std())

# display_scores(lr_rmse_scores)


tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_final,housing_labels)
tree_predictions = tree_reg.predict(housing_final)
tree_mse = mean_squared_error(housing_labels,tree_predictions)
tree_rmse = np.sqrt(tree_mse)
# print("\nTree Regression \n")
# print(tree_rmse)

scores = cross_val_score(tree_reg,housing_final,housing_labels,
	scoring="neg_mean_squared_error",cv=10)
tree_rmse_scores = np.sqrt(-scores)

#Zero error because overfitting
#better evaluation using cross-validation
# display_scores(tree_rmse_scores)

from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_final,housing_labels)
forest_predictions = forest_reg.predict(housing_final)
forest_mse = mean_squared_error(housing_labels,forest_predictions)
forest_rmse = np.sqrt(forest_mse)
# print("\nForest \n")
# print(forest_rmse)
scores = cross_val_score(forest_reg,housing_final,housing_labels,
	scoring="neg_mean_squared_error",cv=10)
forest_rmse_scores = np.sqrt(-scores)
# display_scores(forest_rmse_scores)

from sklearn.model_selection import GridSearchCV

param_grid = [
				{'n_estimators':[3,10,30],'max_features':[2,4,6,8]},
				{'bootstrap':[False],'n_estimators':[3,10],
				'max_features':[2,3,4]},
			]
forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg,param_grid,cv=5,
	scoring="neg_mean_squared_error")

grid_search.fit(housing_final,housing_labels)
# print(grid_search.best_estimator_)

cvres = grid_search.cv_results_
# for mean_score,params in zip(cvres["mean_test_score"],cvres["params"]):
# 	print(np.sqrt(-mean_score),params)
feature_importances = grid_search.best_estimator_.feature_importances_

extra_attribs = ["bedrooms_per_room"]
cat_one_hot_attribs = list(encoder.classes_)
attributes = total_attribs + extra_attribs + cat_one_hot_attribs
feature_importance = sorted(zip(feature_importances,attributes),reverse=True)
# print(feature_importance)
final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value",axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = num_pipeline1.fit_transform(X_test)
print(X_test_prepared)

# final_prediction = final_model.predict(X_test_prepared)
# final_mse = mean_squared_error(y_test,final_prediction)
# final_rmse = np.sqrt(final_mse)

# print(final_rmse)
