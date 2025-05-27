# app/logger.py
import logging
from typing import Annotated
from fastapi import Depends

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("app.log"),
            logging.StreamHandler()
        ]
    )

def get_logger(name: str = __name__):
    return logging.getLogger(name)

# Create a dependency
LoggerDep = Annotated[logging.Logger, Depends(get_logger)]