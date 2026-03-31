import os
import re
from typing import Any, Dict, List, Tuple, Optional
from ming.tools.base_tools import BaseTool, ToolSchema, ToolParameterSchema

class SurgicalEditTool(BaseTool):
    """
    Tool for precise, surgical edits to files using exact string matching.
    """
    def __init__(self, output_dir: str = "outputs"):
        super().__init__("surgical_edit_tool")
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def get_parameters(self) -> ToolSchema:
        return {
            "description": "Perform precise, surgical edits to a file using exact string matching. Best for fixing specific errors in reports.",
            "when_to_use": "Use this to fix specific typos, incorrect citations, or outdated information in a report without rewriting the entire file.",
            "parameters": [
                {
                    "name": "path",
                    "type": "string",
                    "description": "Relative path to the file to modify.",
                    "required": True
                },
                {
                    "name": "source",
                    "type": "string",
                    "description": "The exact text to find in the file. Must match exactly including whitespace and punctuation.",
                    "required": True
                },
                {
                    "name": "target",
                    "type": "string",
                    "description": "The new text to replace the source with.",
                    "required": True
                },
                {
                    "name": "occurrence",
                    "type": "integer",
                    "description": "Which occurrence to replace (1-based index). Use -1 to replace all occurrences. Default is 1.",
                    "required": False
                }
            ]
        }

    def run(self, path: str, source: str, target: str, occurrence: int = 1) -> str:
        try:
            full_path = os.path.join(self.output_dir, path)
            if not os.path.exists(full_path):
                return f"Error: File '{path}' not found in {self.output_dir}"

            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()

            if source not in content:
                return f"Error: Exact source text not found in '{path}'. Please ensure the source text matches exactly."

            count = content.count(source)
            if occurrence == -1:
                new_content = content.replace(source, target)
                msg = f"Successfully replaced all {count} occurrences."
            elif occurrence < 1:
                return f"Error: occurrence must be >= 1 or -1."
            elif occurrence > count:
                return f"Error: occurrence {occurrence} exceeds total count {count}."
            else:
                parts = content.split(source, occurrence)
                new_content = source.join(parts[:occurrence]) + target + source.join(parts[occurrence:])
                msg = f"Successfully replaced occurrence {occurrence} of {count}."

            with open(full_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            
            return msg

        except Exception as e:
            return f"Error: {str(e)}"

    def preflight_check(self) -> bool:
        return True

    def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, str]:
        if "path" not in parameters:
            return False, "Missing 'path'"
        if "source" not in parameters:
            return False, "Missing 'source'"
        if "target" not in parameters:
            return False, "Missing 'target'"
        return True, ""

class ReadFileTool(BaseTool):
    """
    Tool for reading the content of a file.
    """
    def __init__(self, output_dir: str = "outputs"):
        super().__init__("read_file_tool")
        self.output_dir = output_dir

    def get_parameters(self) -> ToolSchema:
        return {
            "description": "Read the content of a file from the output directory.",
            "when_to_use": "Use this to inspect the current state of a report before applying surgical edits.",
            "parameters": [
                {
                    "name": "path",
                    "type": "string",
                    "description": "Relative path to the file to read.",
                    "required": True
                }
            ]
        }

    def run(self, path: str) -> str:
        try:
            full_path = os.path.join(self.output_dir, path)
            if not os.path.exists(full_path):
                return f"Error: File '{path}' not found."
            with open(full_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"Error: {str(e)}"

    def preflight_check(self) -> bool:
        return True

    def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, str]:
        if "path" not in parameters:
            return False, "Missing 'path'"
        return True, ""
