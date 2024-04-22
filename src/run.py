from zenml.pipelines import pipeline
#from zenml.integrations.mlflow.steps import mlflow_model_deployer_step

from data.data_loading        import load_data
from models.predict_model          import evaluate_model
from features.build_features    import feature_engineering, split_train_test, scale_training_data
from models.train_model            import train_rf
from deploy.deploy_triggers     import deployment_trigger
from training_pipeline           import training_rf_pipeline



def main():
    training = training_rf_pipeline(
        get_data = load_data(),
        feature_engineering = feature_engineering(),
        get_train_test_data = split_train_test(),
        scale_data = scale_training_data(),
        train_model = train_rf(),
        evaluate_model = evaluate_model(),
        deployment_trigger = deployment_trigger()#,
        #model_deployer=mlflow_model_deployer_step(), 
    )
    
    training.run()


if __name__ == '__main__':
    main()