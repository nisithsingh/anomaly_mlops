import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# zenml importing
from zenml.steps         import step, Output, BaseParameters
#from zenml.integrations.mlflow.mlflow_step_decorator import enable_mlflow
from sklearn.base import ClassifierMixin


class ModelConfig(BaseParameters):

    model_name: str = "model"

    model_params = {
        'max_depth': 4, 
        'random_state': 42
    }


#@enable_mlflow
@step(experiment_tracker="mlflow_experiment_tracker", enable_cache=False)
def train_rf(X_train: np.ndarray, y_train: np.ndarray, config: ModelConfig) -> Output(
    model = ClassifierMixin
    ):
    """Training a sklearn RF model."""

    params = config.model_params

    model = RandomForestClassifier(**config.model_params)
    
    model.fit(X_train, y_train)


    # mlflow logging
    mlflow.sklearn.log_model(model,config.model_name)
    # y_train_df = pd.DataFrame(y_train)
    # y_train_df.index = X_train.index
    # train_data = pd.concat([X_train,y_train],axis = 1)
    dataset = dataset = mlflow.data.from_numpy(X_train, targets=y_train) #mlflow.data.from_pandas(train_data)
    mlflow.log_input(dataset, context="training")
    
    for param in params.keys():
        mlflow.log_param(f'{param}', params[param])

        
    return model