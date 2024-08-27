import gradio as gr
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('gemstone.csv')
df=df.drop(labels=['id'],axis=1)
X = df.drop(labels=['price'],axis=1)
Y = df[['price']]
categorical_cols = X.select_dtypes(include='object').columns
numerical_cols = X.select_dtypes(exclude='object').columns
cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']
# Numerical Pipeline
num_pipeline=Pipeline(
    steps=[
    ('imputer',SimpleImputer(strategy='median')),
    ('scaler',StandardScaler())
    ]
)
# Categorigal Pipeline
cat_pipeline=Pipeline(
    steps=[
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('ordinalencoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
    ('scaler',StandardScaler())
    ]
)
preprocessor=ColumnTransformer([
    ('num_pipeline',num_pipeline,numerical_cols),
    ('cat_pipeline',cat_pipeline,categorical_cols)
    ]
)

# Train test split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.30,random_state=30)

X_train=pd.DataFrame(preprocessor.fit_transform(X_train),columns=preprocessor.get_feature_names_out())
X_test=pd.DataFrame(preprocessor.transform(X_test),columns=preprocessor.get_feature_names_out())

def evaluate_model(true, predicted):
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    r2_square = r2_score(true, predicted)
    return mae, rmse, r2_square



## Train multiple models
## Model Ecaluation
models={
    'LinearRegression':LinearRegression(),
    'Lasso':Lasso(),
    'Ridge':Ridge(),
    'ElasticNet':ElasticNet(),
    'DecisionTreeRegressor':DecisionTreeRegressor(),
    'SVR':SVR(),
    'RandomForestRegressor':RandomForestRegressor()
}
trained_model_list=[]
model_list=[]
r2_list=[]

for i in range(len(list(models))):
    model=list(models.values())[i]
    model.fit(X_train,y_train)

    #Make Predictions
    y_pred=model.predict(X_test)

    mae, rmse, r2_square=evaluate_model(y_test,y_pred)

    print(list(models.keys())[i])
    model_list.append(list(models.keys())[i])

    print('Model Training Performance')
    print("RMSE:",rmse)
    print("MAE:",mae)
    print("R2 score",r2_square*100)

    r2_list.append(r2_square)
    
    print('='*35)
    print('\n')

imputer = SimpleImputer(strategy='median')

y_train = imputer.fit_transform(y_train)
y_test = imputer.transform(y_test)

randomforestregressor=RandomForestRegressor()
randomforestregressor.fit(X_train,y_train)

def predict_price(carat, cut, color, clarity, depth, table, x, y, z):
    data = {
        'carat': [carat],
        'depth': [depth],
        'table': [table],
        'x': [x],
        'y': [y],
        'z': [z],
        'cut': [cut],
        'color': [color],
        'clarity': [clarity]
    }

    data_df = pd.DataFrame(data)

    processed_data = preprocessor.transform(data_df)

    price_prediction = randomforestregressor.predict(processed_data)
    return price_prediction[0]

iface = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Number(label="Carat"),
        gr.Dropdown(choices=['Fair', 'Good', 'Very Good','Premium','Ideal'], label="Cut"),
        gr.Dropdown(choices=['D', 'E', 'F', 'G', 'H', 'I', 'J'], label="Color"),
        gr.Dropdown(choices=['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF'], label="Clarity"),
        gr.Number(label="Depth"),
        gr.Number(label="Table"),
        gr.Number(label="X"),
        gr.Number(label="Y"),
        gr.Number(label="Z")        
    ],
    outputs="number",
    title="Diamond Price Prediction",
    description="Enter Diamond Characteristics to Predict its Price"
)

iface.launch()

