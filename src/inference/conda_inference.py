"""
Base class for running inference in separate conda environments
"""
import os
import subprocess
import json
import shutil
from pathlib import Path
from typing import Dict, Any


class CondaInferenceRunner:
    def __init__(
        self,
        env_name: str,
        script_name: str,
        temp_dir: str = "temp",
        restore_original_env_on_error: bool = True,
    ):
        self.env_name = env_name
        self.script_name = script_name
        self.temp_dir = temp_dir
        self.script_path = os.path.join(os.path.dirname(__file__), script_name)
        self.restore_original_env_on_error = restore_original_env_on_error

    @staticmethod
    def _capture_conda_environment() -> Dict[str, Any]:
        tracked_keys = [
            "CONDA_DEFAULT_ENV",
            "CONDA_PREFIX",
            "CONDA_PREFIX_1",
            "CONDA_PROMPT_MODIFIER",
            "CONDA_SHLVL",
            "VIRTUAL_ENV",
            "PATH",
        ]
        return {key: os.environ.get(key) for key in tracked_keys}

    @staticmethod
    def _restore_conda_environment(snapshot: Dict[str, Any]) -> None:
        for key, value in snapshot.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _resolve_conda_executable(self) -> str:
        candidates = []

        conda_exe_env = os.environ.get("CONDA_EXE")
        if conda_exe_env:
            candidates.append(conda_exe_env)

        for executable in ("conda", "mamba", "micromamba"):
            resolved = shutil.which(executable)
            if resolved:
                candidates.append(resolved)

        home = Path.home()
        candidates.extend([
            str(home / "miniconda3" / "condabin" / "conda"),
            str(home / "anaconda3" / "condabin" / "conda"),
            str(home / "mambaforge" / "condabin" / "conda"),
            str(home / "miniforge3" / "condabin" / "conda"),
            "/opt/conda/bin/conda",
            "/opt/conda/condabin/conda",
        ])

        for candidate in candidates:
            if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate

        raise RuntimeError(
            "Could not find a conda executable. Set CONDA_EXE or add conda to PATH."
        )
    
    def run(self, input_data: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
        original_env_snapshot = self._capture_conda_environment()

        # write input data
        os.makedirs(self.temp_dir, exist_ok=True)
        base_name = self.script_name.replace("_inference.py", "").replace(".py", "")
        input_json = os.path.join(self.temp_dir, f"{base_name}_input.json")
        output_json = os.path.join(self.temp_dir, f"{base_name}_output.json")
        
        with open(input_json, 'w') as f:
            json.dump(input_data, f)

        if os.path.exists(output_json):
            os.remove(output_json)
        
        # execute command
        conda_executable = self._resolve_conda_executable()
        cmd = [
            conda_executable,
            "run",
            "-n",
            self.env_name,
            "python",
            self.script_path,
            input_json,
            output_json,
        ]
        
        if verbose:
            print(f"Running {base_name} in conda environment '{self.env_name}'...")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # check for errors
        if result.returncode != 0:
            if self.restore_original_env_on_error:
                self._restore_conda_environment(original_env_snapshot)

            error_msg = f"{base_name} inference failed in conda env '{self.env_name}'"
            if result.stderr:
                error_msg += f"\nStderr: {result.stderr}"
            if result.stdout:
                error_msg += f"\nStdout: {result.stdout}"
            raise RuntimeError(error_msg)
        
        if verbose and result.stdout:
            print(result.stdout)

        if not os.path.exists(output_json):
            raise RuntimeError(
                f"{base_name} inference did not produce output file: {output_json}"
            )
        
        # read output
        with open(output_json, 'r') as f:
            return json.load(f)
