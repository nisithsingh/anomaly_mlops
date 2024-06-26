import pandas as pd

# zenml importing
from zenml.steps         import step, Output

@step
def load_data() -> Output(
    data=pd.DataFrame
    ):

    """Load a dataset."""

    data = pd.read_csv('../data/external/cardio_train_sampled.csv')

    return data