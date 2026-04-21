import sys
import os
import time
from apscheduler.schedulers.blocking import BlockingScheduler
from dotenv import load_dotenv

# .env 환경 변수 로드 (현재 파일 위치 기준으로 루트 폴더의 .env 검색)
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
load_dotenv(env_path)

# 모듈 경로 설정을 위해 실행위치/상위 디렉토리를 path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.logger import logger
from app.agent.scheduler import poll_database

if __name__ == "__main__":
    logger.info("====================================")
    logger.info(" 스마트 데이터 마이그레이션 에이전트 가동")
    logger.info("====================================")
    
    # BlockingScheduler 인스턴스 생성
    scheduler = BlockingScheduler()
    
    # 스케줄러에 poll_database 함수를 10초마다 주기적으로 실행하도록 등록합니다.
    # 최초 인스턴스 실행 시 바로 작동 여부 검토를 위해 1초 뒤에 첫 작업이 실행되게 설정할 수 있습니다.
    # 여긴 간단하게 IntervalTrigger로 10초 세팅
    scheduler.add_job(poll_database, 'interval', seconds=10)
    
    logger.info("APScheduler 가동 시작. (10초 주기로 작업 대기열 스캔)")
    logger.info("종료하려면 Ctrl+C 를 누르세요.")
    
    try:
        scheduler.start() # 프로그램이 무한 대기 (백그라운드 스레드에서 주기적 작업 수행)
    except (KeyboardInterrupt, SystemExit):
        logger.info("에이전트가 사용자에 의해 종료되었습니다.")
