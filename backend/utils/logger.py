import sys
import logging
import uuid
from loguru import logger
from config.settings import settings
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

class InterceptHandler(logging.Handler):
    """
    Default handler from examples in loguru documentation.
    Intercept standard logging messages toward loguru sinks.
    """
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
            
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

def setup_logging():
    logger.remove()
    
    log_level = settings.LOG_LEVEL.upper()
    
    # Simple heuristic for development mode vs production
    is_production = settings.USE_ONNX
    
    if is_production:
        logger.add(
            sys.stdout, 
            level=log_level,
            serialize=True
        )
    else:
        logger.add(
            sys.stdout, 
            level=log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level> | {extra}"
        )

    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    # Intercept popular loggers
    for _log in ['uvicorn', 'uvicorn.error', 'uvicorn.access', 'fastapi']:
        _logger = logging.getLogger(_log)
        _logger.handlers = [InterceptHandler()]

def get_logger(name: str):
    return logger.bind(module=name)

class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        with logger.contextualize(request_id=request_id):
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response
