import sys
from loguru import logger
from .config import settings


def setup_logging():
    logger.remove()
    
    if settings.log_format == "json":
        logger.add(
            sys.stdout,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
            level=settings.log_level,
            serialize=True
        )
    else:
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | {message}",
            level=settings.log_level
        )
    
    # File logging
    settings.logs_dir.mkdir(exist_ok=True)
    logger.add(
        settings.logs_dir / "app.log",
        rotation="100 MB",
        retention="30 days",
        level=settings.log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
    )
    
    return logger