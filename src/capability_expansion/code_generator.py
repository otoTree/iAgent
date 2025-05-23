# src/capability_expansion/code_generator.py
import ast
import inspect
from typing import Dict, Any, Callable, Optional # Added Optional
import subprocess
import venv
import tempfile
import os
from pathlib import Path
import json # Added json
import uuid # Added uuid
# Assuming EventBus and Event will be imported from src.event_system
from src.event_system.event_bus import EventBus, Event # Adjusted import
# Assuming memory_system will be passed an instance of InfiniteMemorySystem or similar
# from src.memory.infinite_memory import InfiniteMemorySystem # This might be an import or just a type hint

class CapabilityExpander:
    """自动能力扩展器"""
    def __init__(self, memory_system: Any, event_bus: EventBus): # Added type hint for memory_system
        self.memory_system = memory_system
        self.event_bus = event_bus
        self.generated_tools: Dict[str, Callable] = {}
        self.tool_repository = Path("./generated_tools")
        self.tool_repository.mkdir(exist_ok=True)
        
    async def expand_capability(self, requirement: str) -> Dict[str, Any]:
        """基于需求自动生成新能力"""
        # 1. 分析需求
        analysis = await self._analyze_requirement(requirement)
        
        # 2. 检查是否已有类似能力
        existing = await self._search_existing_capability(analysis)
        if existing:
            return {"status": "exists", "tool": existing}
            
        # 3. 生成代码
        code = await self._generate_tool_code(analysis)
        if not code: # Added check if code generation failed
            return {"status": "code_generation_failed", "reason": "LLM did not return code."}
            
        # 4. 安全验证
        if not await self._validate_code_safety(code):
            return {"status": "unsafe", "reason": "Code failed safety check"}
            
        # 5. 创建隔离环境并测试
        test_result = await self._test_in_sandbox(code, analysis, analysis.get("example_input", {"test_input": "test"})) # Pass analysis and example input
        if not test_result["success"]:
            return {"status": "test_failed", "errors": test_result["errors"], "output": test_result.get("output")}
            
        # 6. 部署新能力
        tool_info = await self._deploy_capability(code, analysis)
        if not tool_info: # Check if deployment failed
             return {"status": "deployment_failed", "reason": "Failed to load or store the new tool."}

        # 7. 更新记忆系统
        await self._update_memory(tool_info)
        
        # 8. 发送能力扩展事件
        event = Event(
            type="capability.expanded",
            data={"tool_name": tool_info["name"], "description": tool_info["description"]},
            source="capability_expander"
        )
        await self.event_bus.emit(event)
        
        return {"status": "success", "tool": tool_info}

    async def _analyze_requirement(self, requirement: str) -> Dict[str, Any]:
        """Placeholder: Analyzes the requirement to extract necessary info for code generation."""
        print(f"Placeholder: Analyzing requirement - {requirement}")
        # In a real system, this would involve LLM calls or complex NLP.
        # For now, returning a dummy analysis.
        return {
            "class_name": "MyNewTool",
            "description": requirement,
            "init_code": "self.message = 'Tool initialized'",
            "execution_code": "result = f'Executing with {kwargs} and {self.message}'",
            "docstring": f"This tool processes the input based on: {requirement}",
            "example_input": {"data": "sample_data_for_MyNewTool"} # Added example input
        }

    async def _search_existing_capability(self, analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Placeholder: Searches memory system for existing tools."""
        print(f"Placeholder: Searching for existing capability matching: {analysis.get('class_name')}")
        # This would query self.memory_system (e.g., skill memory)
        return None 

    async def _generate_tool_code(self, analysis: Dict[str, Any]) -> Optional[str]: # Return Optional[str]
        """生成工具代码"""
        template = """
import asyncio
from typing import Dict, Any
# Assuming BaseTool and @tool decorator are in src.tools.base
from src.tools.base import BaseTool, tool 

@tool
class {class_name}(BaseTool):
    '''{description}'''
    
    def __init__(self):
        super().__init__()
        {init_code}
        
    async def execute(self, **kwargs) -> Dict[str, Any]:
        '''{docstring}'''
        try:
            result = None # Ensure result is initialized
            {execution_code}
            return {{
                "success": True,
                "result": result
            }}
        except Exception as e:
            return {{
                "success": False,
                "error": str(e)
            }}
"""
        
        prompt = f"""
        Generate Python code for a tool with the following requirements:
        Analysis: {analysis}
        
        The code should fill these template variables:
        - class_name: {analysis.get('class_name', 'DefaultToolName')}
        - description: {analysis.get('description', 'Default tool description')}
        - init_code: {analysis.get('init_code', '# No specific init code provided')}
        - docstring: {analysis.get('docstring', 'Default docstring for execute method')}
        - execution_code: {analysis.get('execution_code', 'pass # Default execution logic')}
        
        Return only the Python code string that fills the template variables based on the analysis.
        The code should be directly usable to format the provided template.
        Ensure 'result' is assigned within the execution_code.
        Example of execution_code: result = f"Processed: {{kwargs}}"
        """
        
        print(f"Placeholder: Calling LLM for code generation with prompt:\n{prompt}")
        # generated_code_vars = await self._call_llm_for_code_generation(prompt) # Actual call
        
        # Placeholder LLM response (simulating what the LLM should return)
        generated_code_vars = {
            "class_name": analysis.get('class_name', 'DefaultGeneratedTool'),
            "description": analysis.get('description', 'Generated tool description.'),
            "init_code": analysis.get('init_code', 'pass'),
            "docstring": analysis.get('docstring', 'Executes the generated tool.'),
            "execution_code": analysis.get('execution_code', 'result = "Default execution result"')
        }

        if not generated_code_vars: # Check if LLM call failed
            print("Placeholder: LLM code generation failed to return variables.")
            return None

        try:
            return template.format(**generated_code_vars)
        except KeyError as e:
            print(f"Error formatting tool code template: Missing key {e}")
            return None

    async def _call_llm_for_code_generation(self, prompt: str) -> Optional[Dict[str, str]]:
        """Placeholder: Simulates calling an LLM to get code parts."""
        print(f"Placeholder LLM call for code generation. Prompt: {prompt[:100]}...")
        # This would interact with an actual LLM.
        # For now, returning a fixed dictionary based on a hypothetical analysis.
        # This part needs to be robust in a real system.
        # It should return a dictionary with keys like "class_name", "description", etc.
        # For this placeholder, this method is called by _generate_tool_code which now
        # directly constructs the variables. So this method can just return a dummy dict.
        return {
            "class_name": "PlaceholderLLMTool",
            "description": "Description from LLM.",
            "init_code": "self.llm_initialized = True",
            "docstring": "Docstring from LLM.",
            "execution_code": "result = f'LLM generated execution for {kwargs}'"
        }

    async def _validate_code_safety(self, code: str) -> bool:
        """Placeholder: Validates the generated code for safety."""
        print(f"Placeholder: Validating code safety for:\n{code[:200]}...")
        # In a real system, this would use static analysis, sandboxing, or other checks.
        # For example, check for restricted module imports, dangerous functions, etc.
        if "subprocess.run" in code or "os.system" in code: # A very basic check
             # More sophisticated checks needed, e.g., allow specific subprocess calls if sandboxed
             # For now, this is a simple example.
             # print("Warning: Potential unsafe code detected (subprocess.run or os.system).")
             # return False 
             pass # Allowing for now as the test runner itself uses subprocess
        try:
            ast.parse(code)
        except SyntaxError:
            print("Code safety check failed: Syntax Error.")
            return False
        return True

    async def _test_in_sandbox(self, code: str, analysis: Dict[str, Any], example_input: Dict[str, Any]) -> Dict[str, Any]: # Added analysis
        """在沙箱环境中测试代码"""
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            
            venv_dir = temp_dir / "venv_test"
            try:
                print(f"Creating virtual environment in: {venv_dir}")
                venv.create(venv_dir, with_pip=True, system_site_packages=False) # Explicitly no system site packages
            except Exception as e:
                return {"success": False, "errors": f"Failed to create venv: {e}"}

            # Path to the python executable in the venv
            python_executable = str(venv_dir / "bin" / "python")
            if not os.path.exists(python_executable): # Windows check
                 python_executable = str(venv_dir / "Scripts" / "python.exe")


            # Write the generated tool code
            tool_code_path = temp_dir / "test_tool.py"
            tool_code_path.write_text(code)

            # Write the base tool and tool decorator (minimal version for testing)
            base_tool_code = """
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseTool(ABC):
    def __init__(self):
        pass # Minimal init
    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        pass

def tool(cls): # Minimal decorator
    return cls
"""
            # This base_tool.py needs to be in a place the test_runner.py can import,
            # so placing it inside a 'src/tools' structure within the temp_dir.
            src_tools_dir = temp_dir / "src" / "tools"
            src_tools_dir.mkdir(parents=True, exist_ok=True)
            (src_tools_dir / "base.py").write_text(base_tool_code)
            (temp_dir / "src" / "__init__.py").write_text("") # Make src a package
            (src_tools_dir / "__init__.py").write_text("") # Make src.tools a package
            
            # Create test runner script
            # Ensure that the example_input is correctly formatted as a string for the script
            example_input_str = repr(example_input) # Use repr for a string representation of the dict

            test_script_content = f"""
import sys
import os
import inspect # Added inspect
# Add the root of the temp_dir to sys.path so 'src.tools.base' and 'test_tool' can be found
sys.path.insert(0, r'{str(temp_dir)}')
# print(f"Python version for sandbox: {{sys.version}}")
# print(f"sys.path in sandbox: {{sys.path}}")
# print(f"Current working dir in sandbox: {{os.getcwd()}}")
# print(f"Contents of temp_dir: {{os.listdir(r'{str(temp_dir)}')}}")
# print(f"Contents of src_dir: {{os.listdir(r'{str(temp_dir / 'src')}')}}")
# print(f"Contents of src_tools_dir: {{os.listdir(r'{str(temp_dir / 'src' / 'tools')}')}}")

from test_tool import * 
import asyncio
import json # For potential output parsing

# Dynamically get the class name assuming it's the only class or a specific one
# This is a bit fragile; relies on tool being the main class in test_tool.py
tool_class_name = None
for name, obj in inspect.getmembers(sys.modules['test_tool']):
    if inspect.isclass(obj) and hasattr(obj, 'execute') and name != 'BaseTool': # Check for execute and not BaseTool
        # A more robust check might be needed if there are multiple classes
        if name == "{analysis.get('class_name', 'DefaultToolName')}": # Prioritize expected class name
             tool_class_name = name
             break
        elif tool_class_name is None: # Fallback if expected name not found yet
             tool_class_name = name


async def test():
    if tool_class_name is None:
        print(json.dumps({{"success": False, "error": "Could not find tool class in test_tool.py"}}))
        return

    ToolClass = getattr(sys.modules['test_tool'], tool_class_name)
    tool_instance = ToolClass()
    
    # Use the example_input provided from the analysis
    test_kwargs = {example_input_str}
    
    # print(f"Executing tool {{tool_class_name}} with kwargs: {{test_kwargs}}")
    try:
        result = await tool_instance.execute(**test_kwargs)
        print(json.dumps(result)) # Ensure output is JSON serializable
    except Exception as e:
        print(json.dumps({{"success": False, "error": f"Exception during tool execution: {{str(e)}}"}}))

if __name__ == "__main__":
    asyncio.run(test())
"""
            test_runner_path = temp_dir / "test_runner.py"
            test_runner_path.write_text(test_script_content)
            
            try:
                print(f"Running sandbox test: {python_executable} {test_runner_path}")
                process = subprocess.run(
                    [python_executable, str(test_runner_path)],
                    capture_output=True,
                    text=True,
                    timeout=60, # Increased timeout
                    cwd=str(temp_dir) # Set CWD to allow relative imports within sandbox if necessary
                )
                
                # Try to parse output as JSON
                try:
                    output_json = json.loads(process.stdout)
                except json.JSONDecodeError:
                    output_json = {"raw_stdout": process.stdout}

                return {{
                    "success": process.returncode == 0 and output_json.get("success", False), # Check internal success flag too
                    "output": output_json,
                    "errors": process.stderr if process.returncode != 0 else output_json.get("error")
                }}
            except subprocess.TimeoutExpired:
                return {{
                    "success": False, 
                    "errors": "Test execution timeout (60s)",
                    "output": None
                }}
            except Exception as e:
                 return {{
                    "success": False, 
                    "errors": f"Sandbox execution failed: {str(e)}",
                    "output": None
                }}

    async def _deploy_capability(self, code: str, analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Placeholder: Deploys the new capability (e.g., save to file, load into memory)."""
        tool_name = analysis.get("class_name", f"GeneratedTool_{uuid.uuid4().hex[:6]}")
        file_path = self.tool_repository / f"{tool_name.lower()}.py"
        
        try:
            file_path.write_text(code)
            print(f"Placeholder: Deployed new capability '{tool_name}' to {file_path}")
            
            # Here you might dynamically load the tool or register it
            # For now, just returning info
            tool_info = {
                "name": tool_name,
                "description": analysis.get("description", "N/A"),
                "file_path": str(file_path),
                "code": code # Storing code for potential future use/inspection
            }
            self.generated_tools[tool_name] = tool_info # Store info, not callable directly here
            return tool_info
        except Exception as e:
            print(f"Error deploying capability {tool_name}: {e}")
            return None


    async def _update_memory(self, tool_info: Dict[str, Any]):
        """Placeholder: Updates the memory system with the new capability."""
        print(f"Placeholder: Updating memory system with new tool: {tool_info.get('name')}")
        # Example: self.memory_system.store("skill", tool_info)
        # This depends on the InfiniteMemorySystem's API
        if hasattr(self.memory_system, 'store'):
             await self.memory_system.store(
                 memory_type="skill", 
                 data={
                     "name": tool_info.get("name"),
                     "description": tool_info.get("description"),
                     "code": tool_info.get("code"), # If storing code in memory
                     "file_path": tool_info.get("file_path"), # If storing path
                     # Add other relevant metadata like parameters, usage examples etc.
                 }
            )
        else:
            print(f"Warning: memory_system does not have a 'store' method. Tool '{tool_info.get('name')}' not stored in memory system.")

```
