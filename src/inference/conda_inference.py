"""
Base class for running inference in separate conda environments
"""
import os
import subprocess
import json
import tempfile
from typing import Dict, Any


class CondaInferenceRunner:
    def __init__(self, env_name: str, script_name: str, temp_dir: str = "temp"):
        self.env_name = env_name
        self.script_name = script_name
        self.temp_dir = temp_dir
        self.script_path = os.path.join(os.path.dirname(__file__), script_name)
    
    def run(self, input_data: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
        # write input data
        os.makedirs(self.temp_dir, exist_ok=True)
        base_name = self.script_name.replace("_inference.py", "").replace(".py", "")
        input_json = os.path.join(self.temp_dir, f"{base_name}_input.json")
        output_json = os.path.join(self.temp_dir, f"{base_name}_output.json")
        
        with open(input_json, 'w') as f:
            json.dump(input_data, f)
        
        # execute command
        cmd = f"PYTHONWARNINGS=ignore conda run --no-capture-output -n {self.env_name} python {self.script_path} {input_json} {output_json}"
        
        if verbose:
            print(f"Running {base_name} in conda environment '{self.env_name}'...")
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        # check for errors
        if result.returncode != 0:
            error_msg = f"{base_name} inference failed in conda env '{self.env_name}'"
            if result.stderr:
                error_msg += f"\nStderr: {result.stderr}"
            if result.stdout:
                error_msg += f"\nStdout: {result.stdout}"
            # raise RuntimeError(error_msg)
            print(error_msg)
        
        if verbose and result.stdout:
            print(result.stdout)
        
        # read output
        with open(output_json, 'r') as f:
            return json.load(f)
