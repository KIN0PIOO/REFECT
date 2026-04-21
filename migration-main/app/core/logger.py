import logging
import sys

def setup_logger():
    logger = logging.getLogger("migration_agent")
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        # Windows 등에서 인코딩 문제 방지를 위해 utf-8 설정 시도
        try:
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8', line_buffering=True)
        except:
            pass
            
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - [%(name)s] [%(levelname)s] - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger

logger = setup_logger()
