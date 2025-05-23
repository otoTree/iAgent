# src/main.py
from typing import Dict, Any, Optional
import asyncio
import logging # Added for better logging
from datetime import datetime # Added for _get_current_context and _handle_multimodal_input_processed

# Adjusting imports to be relative to the 'src' directory
from .event_system.event_bus import EventBus, Event, EventPriority
from .multimodal.input_processor import MultiModalInputHandler
from .capability_expansion.code_generator import CapabilityExpander
from .memory.infinite_memory import InfiniteMemorySystem
from .network.web_accessor import WebAccessor
from .agents.orchestrator import MasterOrchestrator

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EvolutionaryAgent:
    """自主进化型AI Agent主系统"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        logger.info("Initializing EvolutionaryAgent...")
        
        # Initialize core components
        self.event_bus = EventBus()
        logger.info("EventBus initialized.")

        self.memory_system = InfiniteMemorySystem(config.get('memory', {}))
        logger.info("InfiniteMemorySystem initialized.")

        self.multimodal_handler = MultiModalInputHandler(self.event_bus)
        logger.info("MultiModalInputHandler initialized.")

        self.capability_expander = CapabilityExpander(self.memory_system, self.event_bus)
        logger.info("CapabilityExpander initialized.")

        self.web_accessor = WebAccessor(self.event_bus, self.memory_system)
        logger.info("WebAccessor initialized.")
        
        # Main orchestrator
        # Tools are passed as a dictionary. The keys are friendly names the orchestrator can use.
        # The values are the actual methods or tool instances.
        agent_tools = {
            'web_access': self.web_accessor, # The orchestrator can call web_accessor.fetch_url, .search_web etc.
            'expand_capability': self.capability_expander.expand_capability # Direct method reference
            # Add other tools here as they are developed/integrated
        }
        self.orchestrator = MasterOrchestrator(
            event_bus=self.event_bus,
            memory=self.memory_system,
            tools=agent_tools 
        )
        logger.info("MasterOrchestrator initialized.")
        
        # Register event handlers
        self._register_event_handlers()
        logger.info("Event handlers registered.")
        
    async def start(self):
        """启动Agent系统"""
        logger.info("Starting EvolutionaryAgent system...")
        # Initialize components that require async initialization
        await self.web_accessor.initialize()
        logger.info("WebAccessor async initialization complete.")
        
        # Start event loop (as a background task)
        event_task = asyncio.create_task(self.event_bus.start())
        logger.info("EventBus processing loop started.")
        
        # Start memory maintenance loop (as a background task)
        memory_task = asyncio.create_task(self._memory_maintenance_loop())
        logger.info("Memory maintenance loop started.")
        
        # Start self-evolution loop (as a background task)
        evolution_task = asyncio.create_task(self._self_evolution_loop())
        logger.info("Self-evolution loop started.")
        
        logger.info("EvolutionaryAgent system started successfully. Waiting for tasks to complete.")
        # Keep the main start() method running, or await specific critical tasks if needed
        # For a persistent agent, these tasks would run indefinitely.
        # Using asyncio.gather to keep them running and allow start() to be awaited if desired.
        try:
            await asyncio.gather(event_task, memory_task, evolution_task)
        except asyncio.CancelledError:
            logger.info("Agent tasks cancelled. Shutting down.")
        finally:
            logger.info("Cleaning up agent resources...")
            if hasattr(self.web_accessor, 'close'):
                await self.web_accessor.close()
            # Add other cleanup logic as needed
            logger.info("Agent shutdown complete.")

    async def process_input(self, input_data: Dict[str, Any]):
        """处理用户输入或外部输入"""
        logger.info(f"Received user input: {input_data}")
        # Emit an event for this input, which will be picked up by _handle_user_input
        event = Event(
            type="user.input.received", # Changed type to be more specific
            data=input_data,
            priority=EventPriority.HIGH,
            source="user_or_external" # General source
        )
        await self.event_bus.emit(event)
        
    def _register_event_handlers(self):
        """注册事件处理器"""
        # Input processing chain
        self.event_bus.subscribe("user.input.received", self._handle_user_input_received) # Renamed handler
        self.event_bus.subscribe("input.text.processed", self._handle_multimodal_input_processed) # Example for text
        self.event_bus.subscribe("input.image.processed", self._handle_multimodal_input_processed) # Example for image
        self.event_bus.subscribe("input.audio.processed", self._handle_multimodal_input_processed) # Example for audio
        self.event_bus.subscribe("input.video.processed", self._handle_multimodal_input_processed) # Example for video
        # Generic handler for any processed input if specific ones don't catch it or for common logic
        self.event_bus.subscribe("input.*.processed", self._handle_any_processed_input) 


        # Capability related events
        # self.event_bus.subscribe("capability.required", self._handle_capability_requirement) # Orchestrator might emit this
        self.event_bus.subscribe("capability.expanded", self._handle_new_capability_expanded) # Renamed handler
        
        # Error handling events (examples)
        self.event_bus.subscribe("error.input.unsupported_modality", self._handle_error_event)
        self.event_bus.subscribe("error.input.processing.*", self._handle_error_event)
        self.event_bus.subscribe("error.memory_maintenance", self._handle_error_event)
        self.event_bus.subscribe("error.*", self._handle_generic_error_event) # Catch-all for other errors

        logger.info("Core event handlers registered.")
        
    async def _handle_user_input_received(self, event: Event):
        """Handles the initial reception of user input."""
        logger.info(f"Handling user.input.received event: {event.data}")
        # Delegate to MultiModalInputHandler
        await self.multimodal_handler.process_input(event.data)
        
    async def _handle_multimodal_input_processed(self, event: Event):
        """Handles events from MultiModalInputHandler after specific modality processing."""
        logger.info(f"Handling {event.type} event with data: {event.data.get('type', 'N/A')}")
        
        processed_data = event.data
        current_context = await self._get_current_context() # Get current agent context

        # Store the processed input into short-term memory
        # This is a more structured storage than the original _handle_processed_input
        memory_entry = {
            "type": "processed_input",
            "modality": processed_data.get("type"),
            "content_summary": str(processed_data)[:200], # Brief summary
            "full_data": processed_data, # The actual processed data
            "received_timestamp": event.timestamp.isoformat(),
            "agent_context": current_context
        }
        memory_id = await self.memory_system.store(
            memory_type="short_term", 
            data=memory_entry,
            # Custom expiry for processed inputs, e.g. 1 hour
            # This needs to be handled by the memory store method if it supports custom TTLs per entry
            # For the example memory system, we might pass 'expiry_hours': 1
        )
        logger.info(f"Processed input stored to short-term memory with ID: {memory_id}")
        
        # Now, send to orchestrator for decision making
        logger.info("Forwarding processed data to MasterOrchestrator...")
        orchestrator_response = await self.orchestrator.process(processed_data)
        logger.info(f"Orchestrator response: {orchestrator_response}")
        
        # Further actions based on orchestrator_response can be handled here or by other events
        # For example, if orchestrator decided on an external action, that might emit another event.

    async def _handle_any_processed_input(self, event: Event):
        """A catch-all or common handler for any 'input.*.processed' event if needed."""
        # This can be used for logging, or if there's common logic after modality-specific handling
        # that isn't covered by _handle_multimodal_input_processed.
        # Ensure this doesn't duplicate actions if more specific handlers already did the work.
        # For now, just logging.
        logger.debug(f"Generic input processed event caught: {event.type}")


    async def _handle_new_capability_expanded(self, event: Event):
        """Handles the event that a new capability has been successfully expanded/added."""
        tool_name = event.data.get("tool_name")
        description = event.data.get("description")
        logger.info(f"New capability expanded: '{tool_name}' - '{description}'.")
        # The CapabilityExpander already updated memory and deployed the tool.
        # The Orchestrator might need to be made aware of new tools dynamically if its toolset is static.
        # For simplicity, if tools are dynamically loaded by Orchestrator or if it re-queries, this might be fine.
        # Otherwise, one might need to update self.orchestrator.tools here (if safe and designed for it).
        # For now, we assume the CapabilityExpander handles making the tool available.
        pass

    async def _handle_error_event(self, event: Event):
        """Handles specific error events."""
        logger.error(f"Specific error event received: Type='{event.type}', Data='{event.data}', Source='{event.source}'")
        # Potential actions: log to a dedicated error store, notify admin, attempt recovery.

    async def _handle_generic_error_event(self, event: Event):
        """Handles generic error events not caught by more specific handlers."""
        # Avoid logging events that were already handled by _handle_error_event if it's also subscribed to error.*
        if event.type not in ["error.input.unsupported_modality", "error.memory_maintenance"] and not event.type.startswith("error.input.processing"):
             logger.error(f"Generic error event received: Type='{event.type}', Data='{event.data}', Source='{event.source}'")


    async def _get_current_context(self) -> Dict[str, Any]:
        """Placeholder: Retrieves the current operational context of the agent."""
        # This could include current task, user session, focus, recent interactions, etc.
        return {
            "current_time": datetime.now().isoformat(),
            "active_task_id": "task_001", # Example
            "user_sentiment": "neutral" # Example
        }

    async def _memory_maintenance_loop(self):
        """Background loop for periodic memory maintenance tasks."""
        while True:
            await asyncio.sleep(self.config.get('memory_maintenance_interval', 3600)) # Default 1 hour
            logger.info("Starting scheduled memory maintenance...")
            try:
                if hasattr(self.memory_system, '_consolidate_memories'):
                    await self.memory_system._consolidate_memories()
                if hasattr(self.memory_system, '_clean_expired_memories'): # If method exists
                    await self.memory_system._clean_expired_memories()
                if hasattr(self.memory_system, '_optimize_knowledge_graph'): # If method exists
                    await self.memory_system._optimize_knowledge_graph()
                logger.info("Memory maintenance finished.")
            except Exception as e:
                logger.error(f"Error during memory maintenance: {e}")
                await self.event_bus.emit(Event(
                    type="error.memory_maintenance",
                    data={"error": str(e)},
                    priority=EventPriority.LOW,
                    source="EvolutionaryAgent"
                ))
                
    async def _self_evolution_loop(self):
        """Background loop for agent's self-evolution and improvement tasks."""
        while True:
            await asyncio.sleep(self.config.get('self_evolution_interval', 86400)) # Default 1 day
            logger.info("Starting self-evolution cycle...")
            try:
                # 1. Analyze performance (placeholder)
                metrics = await self._analyze_performance_placeholder()
                
                # 2. Identify improvement opportunities (placeholder)
                improvements = await self._identify_improvements_placeholder(metrics)
                
                # 3. Implement improvements (placeholder)
                for improvement in improvements:
                    logger.info(f"Attempting improvement: {improvement}")
                    if improvement.get('type') == 'new_capability' and 'requirement' in improvement:
                        # Use the capability expander tool via its direct method
                        await self.capability_expander.expand_capability(improvement['requirement'])
                    elif improvement.get('type') == 'optimize_prompt':
                        # await self._optimize_prompts_placeholder(improvement.get('target'))
                        logger.info(f"Placeholder: Optimizing prompts for {improvement.get('target')}")
                    elif improvement.get('type') == 'adjust_parameters':
                        # await self._adjust_parameters_placeholder(improvement.get('params'))
                        logger.info(f"Placeholder: Adjusting parameters {improvement.get('params')}")
                logger.info("Self-evolution cycle finished.")
            except Exception as e:
                logger.error(f"Error during self-evolution cycle: {e}")
                await self.event_bus.emit(Event(
                    type="error.self_evolution",
                    data={"error": str(e)},
                    priority=EventPriority.LOW,
                    source="EvolutionaryAgent"
                ))

    async def _analyze_performance_placeholder(self) -> Dict[str, Any]:
        logger.info("Placeholder: Analyzing agent performance...")
        # Example: query memory for task success rates, tool errors, user feedback
        return {"task_success_rate": 0.85, "common_errors": ["tool_timeout"], "new_requests_unhandled": 5}

    async def _identify_improvements_placeholder(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        logger.info(f"Placeholder: Identifying improvements based on metrics: {metrics}")
        improvements = []
        if metrics.get("new_requests_unhandled", 0) > 3:
            improvements.append({
                "type": "new_capability", 
                "requirement": "Develop a tool to handle frequently unhandled request patterns. (Self-identified)"
            })
        if "tool_timeout" in metrics.get("common_errors", []):
             improvements.append({
                "type": "adjust_parameters",
                "params": {"tool_timeout_increase": "some_tool_name"} # Which tool?
             })
        return improvements

# --- Example Usage ---
async def main():
    logger.info("Starting agent configuration...")
    # Example configuration
    # In a real app, this might come from a YAML file or environment variables
    config = {
        "memory": {
            "redis_host": "localhost", # Replace with your Redis host if not localhost
            "redis_port": 6379,
            "neo4j_uri": "bolt://localhost:7687", # Replace if needed
            "neo4j_user": "neo4j",
            "neo4j_password": "password", # Replace with your Neo4j password
            # ChromaDB in InfiniteMemorySystem defaults to in-memory or local path
        },
        "llm": { # This config part is not directly used by current placeholders but good for future
            "provider": "openai", # or "anthropic", "google", etc.
            "model": "gpt-4", 
            "api_key": "YOUR_API_KEY_HERE" # Ensure this is set if LLM calls were active
        },
        "memory_maintenance_interval": 300, # Shortened for testing (5 minutes)
        "self_evolution_interval": 600    # Shortened for testing (10 minutes)
    }
    
    agent = EvolutionaryAgent(config)
    
    # Start the agent's background tasks (event loop, maintenance, evolution)
    # agent.start() will run indefinitely until cancelled.
    agent_instance_task = asyncio.create_task(agent.start())
    logger.info("Agent instance started in background task.")

    # Give the agent a moment to initialize background tasks
    await asyncio.sleep(2) 

    # Example: Process a text input
    logger.info("Sending example text input to agent...")
    await agent.process_input({
        "modality": "text",
        "data": "Please search web for the current weather in London." 
    })
    
    await asyncio.sleep(10) # Let it process and potentially use a tool

    logger.info("Sending another example text input for capability expansion...")
    await agent.process_input({
        "modality": "text",
        "data": "I need a new tool for calculating SHA256 hash of a string."
    })

    # Let the agent run for a bit longer to see maintenance/evolution logs (if intervals are short)
    # In a real deployment, agent_instance_task would be awaited or managed by a supervisor.
    # For this example, we might let it run for a minute then cancel.
    await asyncio.sleep(config["memory_maintenance_interval"] + 5) # Wait for one maintenance cycle
    
    logger.info("Example interactions complete. Main script will now cancel the agent task.")
    agent_instance_task.cancel()
    try:
        await agent_instance_task # Wait for cancellation to complete and cleanup
    except asyncio.CancelledError:
        logger.info("Agent task successfully cancelled from main().")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Main script interrupted by user (Ctrl+C). Shutting down.")
    finally:
        logger.info("Main script finished.")

# For relative imports to work correctly when running this file directly for testing:
# Ensure you are in the directory *above* 'src' and run as 'python -m src.main'
# If running directly like 'python src/main.py', Python might not treat 'src' as a package.
# The __init__.py files in src and subdirectories help, but execution context matters.
```
