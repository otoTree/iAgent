# src/event_system/event_bus.py
from typing import Dict, List, Callable, Any
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
import uuid
from enum import Enum

class EventPriority(Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3

@dataclass
class Event:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    priority: EventPriority = EventPriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

class EventBus:
    """核心事件流引擎"""
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._event_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._event_history: List[Event] = []
        self._running = False
        
    async def emit(self, event: Event):
        """发送事件到事件流"""
        # 添加到优先级队列
        priority_value = -event.priority.value  # 负值使高优先级先处理
        await self._event_queue.put((priority_value, event))
        
    def subscribe(self, event_type: str, handler: Callable):
        """订阅特定类型的事件"""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)
        
    async def start(self):
        """启动事件处理循环"""
        self._running = True
        while self._running:
            try:
                _, event = await self._event_queue.get()
                await self._process_event(event)
            except Exception as e:
                await self._handle_error(e, event) # Add event argument
                
    async def _process_event(self, event: Event):
        """处理单个事件"""
        self._event_history.append(event)
        
        # 通知所有订阅者
        handlers = self._subscribers.get(event.type, [])
        handlers.extend(self._subscribers.get("*", []))  # 全局订阅者
        
        # 并发执行所有处理器
        tasks = [handler(event) for handler in handlers]
        await asyncio.gather(*tasks, return_exceptions=True)

    # Added placeholder for _handle_error as it's called but not defined
    async def _handle_error(self, error: Exception, event: Event):
        print(f"Error processing event {event.id} of type {event.type}: {error}")
        # In a real system, you'd likely emit another event or log to a more robust system
