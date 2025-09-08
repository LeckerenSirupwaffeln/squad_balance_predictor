# Python imports

# 3rd party imports
from loguru import logger

# Relative imports
from src.constants import (
    MATCHES_FILE,
    TRAINING_BALANCE_FILE,
    TRAINING_VOLATILITY_FILE,
    MODEL_BALANCE_FILE,
    MODEL_VOLATILITY_FILE,
    RESULTS_FOLDER
)
from src.download_matches import download_matches
from src.process_matches import process_matches
from src.train_model import train_model
from src.predict_matches import predict_matches

@logger.catch
def main():
    download_matches(matches_file=MATCHES_FILE, pages=2000, page_size=100)
    process_matches(MATCHES_FILE, TRAINING_BALANCE_FILE, TRAINING_VOLATILITY_FILE)
    train_model(
        TRAINING_BALANCE_FILE,
        TRAINING_VOLATILITY_FILE,
        MODEL_BALANCE_FILE,
        MODEL_VOLATILITY_FILE
    )
    predict_matches(MODEL_BALANCE_FILE, MODEL_VOLATILITY_FILE, RESULTS_FOLDER)
    return


if __name__ == "__main__":
    main()
