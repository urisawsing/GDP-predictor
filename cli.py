#!/usr/bin/env python
import sys
import os
import logging
import argparse
from datetime import datetime

sys.path.append('.')

from utils import config, io, models


logging.basicConfig(
    filename=os.path.join(config.LOGS_PATH, datetime.now().strftime('cli_%Y-%m-%d_%H:%M:%S.log')),
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    level=logging.INFO,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "task",
    choices=["train", "predict"],
    help="Task to be performed",
)
parser.add_argument(
    "model",
    choices=["GradBoost", "MultiLin"],
    help="Model used",
)
parser.add_argument(
    "priority",
    choices=["Time", "Results"],
    help="Choosing between fast execution or exploratory execution",
)
    
# You can add here custom optional arguments to your program

if __name__ == "__main__":
    args = parser.parse_args()
    if args.task == "train":
        logging.info("Training")
    
        if args.model =="Gradboost":
            logging.info("Gradient_Boosting_Method")
    
            if args.priority =="Time":
                logging.info("Priorizing_Time")
                R=models.GBmodelTrain()
            
            if args.priority =="Results":
                logging.info("Priorizing_Results")
                models.GBmodelTrain()
                R=models.ExhaustiveGBM()
    
        if args.model=="MultiLin":
            logging.info("MultiLinear_Method")
    
            if args.priority =="Time":
                logging.info("Priorizing_Time")
                R=models.multilinearTrain()
    
            if args.priority =="Results":
                logging.info("Priorizing_Results")
                R=models.ExhaustiveML()
            
    
    
    if args.task == "predict":
        logging.info("Predicting")
    
        if args.model =="GradBoost":
            models.GBmodelPredict()
    
        if args.model =="MultiLin":
            models.multilinearPredict()
