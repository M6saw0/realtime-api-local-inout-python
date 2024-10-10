import asyncio
import datetime
import os

import loguru


class Logger:
    def __init__(self, base_dir: str="./log"):
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        self.logger = loguru.logger
        self.logger.remove()
        self.logger.add(
            f"{base_dir}/{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.log",
            format="{time:YYYY-MM-DD HH:mm:ss.SSSSSS ZZ} | {level:5} | {message}",
            # rotation="1 MB",
            encoding="utf-8"
        )

    async def debug(self, message):
        self.logger.debug(message)

    async def info(self, message):
        self.logger.info(message)

    async def error(self, message):
        self.logger.error(message)


async def main(logger: Logger):
    await logger.info("Hello, World!")
    await logger.error("Hello, World!")
    await logger.debug("Hello, World!")

if __name__ == "__main__":
    logger = Logger()
    asyncio.run(main(logger))
