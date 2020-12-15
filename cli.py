#!/usr/bin/env python
import sys
import os
import logging
import argparse
from datetime import datetime

sys.path.append('.')

from utils import config, models
from utils import exhaustive_methods as exh

logging.basicConfig(
    level=logging.INFO,
    filename=os.path.join(config.LOGS_PATH, datetime.now().strftime('cli_%Y-%m-%d_%H.%M.%S.log')),
    format="%(asctime)s - [%(levelname)s] - %(message)s"
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "task",
    choices=["train", "predict"],
    help="Task to be performed"
)
parser.add_argument(
    "model",
    choices=["GB", "ML"],
    default="GB",
    help="Model used, gradientboost or multilinear"
)
parser.add_argument(
    "priority",
    choices=["T", "R"],
    default="T",
    help="Choosing between fast execution(time) or exploratory execution(results)"
)

# You can add here custom optional arguments to your program

if __name__ == "__main__":
    args = parser.parse_args()
    if args.task == "train":
        logging.info("Training")

        if args.model == "GB":
            logging.info("Gradient_Boosting_Method")

            if args.priority == "T":
                logging.info("Priorizing_Time")
                R = models.GBmodelTrain()
            elif args.priority == "R":
                msg = f"Priorizing_Results with {config.EXHAUSTIVE_ITER} iterations"
                logging.info(msg)
                R = exh.ExhaustiveGBM()
        elif args.model == "ML":
            logging.info("MultiLinear_Method")

            if args.priority == "T":
                logging.info("Priorizing_Time")
                R = models.multilinearTrain()
            elif args.priority == "R":
                msg = f"Priorizing_Results with {config.EXHAUSTIVE_ITER} iterations"
                logging.info(msg)
                R = exh.ExhaustiveML()
        logging.info(R)
    elif args.task == "predict":
        logging.info("Predicting")

        if args.model == "GB":
            models.GBmodelPredict()
        elif args.model == "ML":
            models.multilinearPredict()
