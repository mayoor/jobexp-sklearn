import sklearn
import pandas as pd
import joblib
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import os


def train(logger=None, **kwargs):
    if not logger:
        import logging
        logger = logging.getlogger("mayoor.sklearn.trainer")
    boston_data = load_boston()
    df = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)
    target = boston_data.target
    rf_model = RandomForestRegressor(n_jobs=-1, max_depth=5, random_state=42)
    xtrain, xtest, ytrain, ytest = train_test_split(
        df, target, test_size=0.2, random_state=42
    )
    rf_model.fit(xtrain, ytrain)
    logger.log(f"Test score is: {r2_score(ytest, rf_model.predict(xtest))}")
    output_dir = os.environ["OUTPUT_DIR"]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, "model.jlib"), "wb") as mf:
        joblib.dump(rf_model, mf)


def test(logger=None, **kwargs):
    if not logger:
        import logging
        logger = logging.getlogger("mayoor.sklearn.tester")
    logger.log("I am testing")


def echo(name, logger=None, **kwargs):
    logger.log(f"Echo: {name}")
