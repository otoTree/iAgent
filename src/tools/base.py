# src/tools/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Callable, Optional
import inspect
import asyncio # Added for async execute

# Global registry for tools
TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {}

class BaseTool(ABC):
    """
    工具基类 (Base Class for Tools)
    All tools should inherit from this class.
    """
    def __init__(self):
        # print(f"Tool '{self.__class__.__name__}' initialized.") # Commented out as requested
        self.name: str = self.__class__.__name__ # Tool name is class name
        self.description: str = inspect.getdoc(self) or "No description provided." # From class docstring
        self.parameters: List[Dict[str, Any]] = self._extract_parameters()

    def _extract_parameters(self) -> List[Dict[str, Any]]:
        """
        Extracts parameters for the 'execute' method using introspection.
        Assumes 'execute' method has type hints and a docstring.
        """
        params_list = []
        if hasattr(self, 'execute') and callable(self.execute):
            sig = inspect.signature(self.execute)
            # Parse docstring for parameter descriptions (very basic parsing)
            docstring = inspect.getdoc(self.execute)
            param_docs = {}
            if docstring:
                lines = docstring.split('\n')
                for line in lines:
                    if ":" in line and "Args:" not in line and "Returns:" not in line: # Simple check
                        parts = line.strip().split(":", 1)
                        if len(parts) == 2:
                             p_name = parts[0].strip()
                             p_desc = parts[1].strip()
                             # Remove potential type hints from Sphinx-style docstrings
                             if '(' in p_name and ')' in p_name:
                                 p_name = p_name.split('(')[0].strip()
                             param_docs[p_name] = p_desc
            
            for name, param in sig.parameters.items():
                if name == 'self' or name == 'kwargs': # Skip self and general kwargs
                    continue
                
                param_info = {
                    "name": name,
                    "type": str(param.annotation) if param.annotation != inspect.Parameter.empty else "Any",
                    "required": param.default == inspect.Parameter.empty,
                    "description": param_docs.get(name, "No description.") # Get from parsed docstring
                }
                if param.default != inspect.Parameter.empty:
                    param_info["default"] = param.default
                params_list.append(param_info)
        return params_list

    @abstractmethod
    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Executes the tool's main functionality.
        Must be implemented by subclasses.
        Should return a dictionary with results.
        
        Example Args:
            parameter_name (type): Description of the parameter.
            
        Returns:
            Dict[str, Any]: A dictionary containing the execution result.
                            Typically includes 'success': True/False and other data.
        """
        pass

    # Optional helper methods common to tools
    async def _emit_event(self, event_bus: Any, event_type: str, data: Dict[str, Any]):
        """Helper to emit events if an event bus is available."""
        if event_bus and hasattr(event_bus, 'emit'):
            from src.event_system.event_bus import Event # Local import to avoid circular dependency issues at module load time
            await event_bus.emit(Event(type=event_type, data=data, source=self.name))
        else:
            print(f"Event bus not available or 'emit' method missing. Event '{event_type}' not sent from tool '{self.name}'.")

def tool(cls: type) -> type:
    """
    工具注册装饰器 (Decorator to register a tool)
    Registers the decorated class in the TOOL_REGISTRY.
    """
    if not issubclass(cls, BaseTool):
        raise TypeError("Decorated class must be a subclass of BaseTool.")

    tool_name = cls.__name__
    # print(f"Registering tool: {tool_name}") # Optional: for debugging registration

    # Instantiate to get parameters, but don't store instance if not needed
    try:
        temp_instance = cls()
        tool_info = {
            "class": cls,
            "name": tool_name,
            "description": temp_instance.description,
            "parameters": temp_instance.parameters, # Extracted from execute method
            "module": cls.__module__
        }
        TOOL_REGISTRY[tool_name] = tool_info
    except Exception as e:
        print(f"Error during registration of tool '{tool_name}': {e}. Tool may not be fully registered or functional.")
        # Register with minimal info if instantiation fails (e.g. constructor needs args not available at decoration time)
        TOOL_REGISTRY[tool_name] = {
            "class": cls,
            "name": tool_name,
            "description": inspect.getdoc(cls) or "Error during description retrieval.",
            "parameters": [], # Cannot get params if init fails
            "module": cls.__module__,
            "registration_error": str(e)
        }
    
    return cls

# --- Example Usage (Commented out as per request) ---
# @tool
# class MyExampleTool(BaseTool):
#     """
#     This is a simple example tool.
#     It demonstrates how to define a tool using BaseTool and @tool decorator.
#     """
#     def __init__(self):
#         super().__init__()
#         # Custom initialization for MyExampleTool
#         self.prefix = "ExampleTool processed: "

#     async def execute(self, text_input: str, number_input: int = 0) -> Dict[str, Any]:
#         """
#         Executes the example tool.

#         Args:
#             text_input (str): The text to process.
#             number_input (int): A number to include in processing. Default is 0.
        
#         Returns:
#             Dict[str, Any]: A dictionary with the processed string and status.
#         """
#         if not text_input:
#             return {"success": False, "error": "text_input cannot be empty."}
        
#         processed_string = f"{self.prefix}{text_input} (Number: {number_input})"
#         return {"success": True, "result": processed_string, "number_used": number_input}

# async def main_test():
#     """For testing the tool registration and execution."""
#     print("Available tools in registry:")
#     for tool_name, info in TOOL_REGISTRY.items():
#         print(f"- {tool_name}: {info.get('description')}")
#         print(f"  Parameters: {info.get('parameters')}")

#     if "MyExampleTool" in TOOL_REGISTRY:
#         ExampleToolClass = TOOL_REGISTRY["MyExampleTool"]["class"]
#         example_tool_instance = ExampleToolClass()
        
#         print("\nExecuting MyExampleTool...")
#         result1 = await example_tool_instance.execute(text_input="Hello World", number_input=42)
#         print(f"Result 1: {result1}")
        
#         result2 = await example_tool_instance.execute(text_input="Another test")
#         print(f"Result 2: {result2}")

#         result_fail = await example_tool_instance.execute(text_input="")
#         print(f"Result Fail: {result_fail}")

# if __name__ == "__main__":
#     # This part is for direct testing of this file.
#     # Note: If tools have dependencies on other parts of the system (like event_bus),
#     # they might not work as expected when run standalone like this without proper mocking.
#     asyncio.run(main_test())
```
