import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from ming.models.openrouter_model import OpenRouterModel, OpenRouterModelConfig
from ming.subagent import Agent, AgentConfig
from ming.tools.surgical_edit_tool import SurgicalEditTool, ReadFileTool
from ming.core.prompts import AUDITOR_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

class QualityAuditor:
    def __init__(self, model_config: OpenRouterModelConfig, output_dir: str = "outputs"):
        self.model_config = model_config
        self.output_dir = output_dir
        
        # Tools owned by the Auditor
        self.surgical_tool = SurgicalEditTool(output_dir=output_dir)
        self.read_tool = ReadFileTool(output_dir=output_dir)
        
        # The Auditor is now a tool-calling Agent
        self.agent = Agent(
            AgentConfig(
                model=model_config,
                system_prompt=AUDITOR_SYSTEM_PROMPT,
                tools=[self.surgical_tool, self.read_tool],
                max_iterations=10, # Give it enough rounds to fix multiple issues
            )
        )

    def audit_and_fix(self, report_filename: str) -> str:
        """
        Audit the report and autonomously fix any issues found.
        Returns the final (potentially fixed) markdown content.
        """
        instruction = f"Please audit and fix the research report located at '{report_filename}'. If you find quality issues like placeholders or missing content, use your tools to apply surgical fixes. If the report is already high quality, just say it passed."
        
        logger.info(f"QualityAuditor: Starting audit and fix for {report_filename}")
        
        try:
            # The agent will now loop: Read -> Identify Issue -> Replace -> Repeat
            result = self.agent.run(instruction)
            
            # Read the final state of the file after the agent's work
            full_path = Path(self.output_dir) / report_filename
            if full_path.exists():
                return full_path.read_text(encoding="utf-8")
            else:
                logger.error(f"QualityAuditor: Report file {report_filename} disappeared!")
                return ""
        except Exception as e:
            logger.error(f"QualityAuditor: Audit and fix loop failed: {e}")
            # Fallback to current file content
            full_path = Path(self.output_dir) / report_filename
            if full_path.exists():
                return full_path.read_text(encoding="utf-8")
            return ""
