from fastapi import HTTPException
import asyncio
from collections import deque
import time
from typing import Any, Callable, Coroutine

class RequestQueue:
    def __init__(self, max_queue_size: int = 10, timeout: int = 30):
        self.queue = deque()
        self.max_queue_size = max_queue_size
        self.timeout = timeout
        self.processing = False

    async def enqueue_request(self, process_func: Callable[..., Coroutine], *args: Any) -> Any:
        if len(self.queue) >= self.max_queue_size:
            raise HTTPException(status_code=503, detail="Queue is full, try again later")

        future = asyncio.Future()
        self.queue.append((future, process_func, args, time.time()))
        
        if not self.processing:
            asyncio.create_task(self._process_queue())

        try:
            return await asyncio.wait_for(future, timeout=self.timeout)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail="Request timed out")

    async def _process_queue(self):
        self.processing = True
        while self.queue:
            future, func, args, start_time = self.queue.popleft()
            if time.time() - start_time > self.timeout:
                future.set_exception(HTTPException(status_code=408, detail="Request timed out"))
                continue

            try:
                result = await func(*args)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)

        self.processing = False