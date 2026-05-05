"""
Base class for running inference in separate conda environments
"""
import os
import subprocess
import json
import shutil
import queue
import threading
import uuid
import glob
import shlex
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

    @staticmethod
    def _resolve_conda_activation_script(conda_executable: str) -> str:
        conda_path = Path(conda_executable).resolve()
        try:
            conda_base = conda_path.parents[1]
        except IndexError as exc:
            raise RuntimeError(
                f"Could not resolve conda base from executable: {conda_executable}"
            ) from exc

        conda_sh = conda_base / "etc" / "profile.d" / "conda.sh"
        if not conda_sh.is_file():
            raise RuntimeError(
                f"Could not find conda activation script at: {conda_sh}"
            )
        return str(conda_sh)
    
    def run(self, input_data: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
        original_env_snapshot = self._capture_conda_environment()

        # write input data
        temp_dir = os.path.abspath(self.temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
        base_name = self.script_name.replace("_inference.py", "").replace(".py", "")
        input_json = os.path.join(temp_dir, f"{base_name}_input.json")
        output_json = os.path.join(temp_dir, f"{base_name}_output.json")
        
        with open(input_json, 'w') as f:
            json.dump(input_data, f)

        if os.path.exists(output_json):
            os.remove(output_json)
        
        # execute command
        conda_executable = self._resolve_conda_executable()
        conda_sh = self._resolve_conda_activation_script(conda_executable)
        cmd = [
            "bash",
            "-lc",
            " ".join([
                f"source {shlex.quote(conda_sh)}",
                f"conda activate {shlex.quote(self.env_name)}",
                "python",
                shlex.quote(self.script_path),
                shlex.quote(input_json),
                shlex.quote(output_json),
            ]),
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
            error_msg = (
                f"{base_name} inference did not produce output file: {output_json}"
            )
            if result.stderr:
                error_msg += f"\nStderr: {result.stderr}"
            if result.stdout:
                error_msg += f"\nStdout: {result.stdout}"
            raise RuntimeError(error_msg)
        
        # read output
        with open(output_json, 'r') as f:
            return json.load(f)


class ThreadedCondaInferenceRunner:
    _RESULT_PREFIX = "__INFER_DONE__"

    def __init__(self, env_name: str, worker_script_name: str, temp_dir: str = "temp"):
        self.env_name = env_name
        self.worker_script_name = worker_script_name
        self.temp_dir = temp_dir
        self.worker_script_path = os.path.join(os.path.dirname(__file__), worker_script_name)

        self._request_queue: "queue.Queue[Any]" = queue.Queue()
        self._thread = None
        self._shutdown = False

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

    @staticmethod
    def _resolve_conda_activation_script(conda_executable: str) -> str:
        conda_path = Path(conda_executable).resolve()
        try:
            conda_base = conda_path.parents[1]
        except IndexError as exc:
            raise RuntimeError(
                f"Could not resolve conda base from executable: {conda_executable}"
            ) from exc

        conda_sh = conda_base / "etc" / "profile.d" / "conda.sh"
        if not conda_sh.is_file():
            raise RuntimeError(
                f"Could not find conda activation script at: {conda_sh}"
            )
        return str(conda_sh)

    def _resolve_env_python_executable(self) -> str:
        conda_executable = self._resolve_conda_executable()
        conda_sh = self._resolve_conda_activation_script(conda_executable)

        probe_cmd = [
            "bash",
            "-lc",
            " ".join([
                f"source {shlex.quote(conda_sh)}",
                f"conda activate {shlex.quote(self.env_name)}",
                "python",
                "-c",
                shlex.quote("import sys; print(sys.executable)"),
            ]),
        ]

        try:
            probe = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
            candidates = [line.strip() for line in probe.stdout.splitlines() if line.strip()]
            if candidates:
                python_path = candidates[-1]
                if os.path.isfile(python_path) and os.access(python_path, os.X_OK):
                    return python_path
        except Exception:
            pass

        try:
            info_cmd = [
                conda_executable,
                "info",
                "--envs",
                "--json",
            ]
            info = subprocess.run(info_cmd, capture_output=True, text=True, check=True)
            info_data = json.loads(info.stdout or "{}")
            envs = info_data.get("envs", [])
            for env_path in envs:
                if os.path.basename(env_path) == self.env_name:
                    candidate = os.path.join(env_path, "bin", "python")
                    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                        return candidate
        except Exception:
            pass

        try:
            base_cmd = [
                conda_executable,
                "info",
                "--base",
            ]
            base = subprocess.run(base_cmd, capture_output=True, text=True, check=True)
            conda_base = base.stdout.strip()
        except Exception:
            conda_base = None

        home = Path.home()
        fallback_candidates = [
            home / "miniconda3" / "envs" / self.env_name / "bin" / "python",
            home / "anaconda3" / "envs" / self.env_name / "bin" / "python",
            home / "mambaforge" / "envs" / self.env_name / "bin" / "python",
            home / "miniforge3" / "envs" / self.env_name / "bin" / "python",
            Path("/opt/conda/envs") / self.env_name / "bin" / "python",
        ]
        if conda_base:
            fallback_candidates.insert(
                0, Path(conda_base) / "envs" / self.env_name / "bin" / "python"
            )
        for candidate in fallback_candidates:
            candidate_str = str(candidate)
            if os.path.isfile(candidate_str) and os.access(candidate_str, os.X_OK):
                return candidate_str

        raise RuntimeError(
            f"Could not resolve python executable for conda env '{self.env_name}'."
        )

    def _build_worker_env(self, env_python: str) -> Dict[str, str]:
        env = os.environ.copy()

        env_root = os.path.dirname(os.path.dirname(env_python))
        candidate_dirs = [
            os.path.join(env_root, "lib"),
            os.path.join(env_root, "lib64"),
            os.path.join(env_root, "targets", "x86_64-linux", "lib"),
        ]

        site_packages_glob = os.path.join(env_root, "lib", "python*", "site-packages")
        site_package_dirs = [p for p in glob.glob(site_packages_glob) if os.path.isdir(p)]

        for site_dir in site_package_dirs:
            candidate_dirs.extend([
                os.path.join(site_dir, "torch", "lib"),
                os.path.join(site_dir, "nvidia", "cuda_nvrtc", "lib"),
                os.path.join(site_dir, "nvidia", "cuda_nvrtc", "lib64"),
                os.path.join(site_dir, "nvidia", "cuda_nvrtc", "targets", "x86_64-linux", "lib"),
                os.path.join(site_dir, "nvidia", "cuda_runtime", "lib"),
                os.path.join(site_dir, "nvidia", "cuda_runtime", "lib64"),
                os.path.join(site_dir, "nvidia", "cuda_runtime", "targets", "x86_64-linux", "lib"),
                os.path.join(site_dir, "nvidia", "cudnn", "lib"),
                os.path.join(site_dir, "nvidia", "cudnn", "lib64"),
                os.path.join(site_dir, "nvidia", "cudnn", "targets", "x86_64-linux", "lib"),
                os.path.join(site_dir, "nvidia", "cublas", "lib"),
                os.path.join(site_dir, "nvidia", "cublas", "lib64"),
                os.path.join(site_dir, "nvidia", "cublas", "targets", "x86_64-linux", "lib"),
                os.path.join(site_dir, "nvidia", "cusolver", "lib"),
                os.path.join(site_dir, "nvidia", "cusolver", "lib64"),
                os.path.join(site_dir, "nvidia", "cusolver", "targets", "x86_64-linux", "lib"),
                os.path.join(site_dir, "nvidia", "curand", "lib"),
                os.path.join(site_dir, "nvidia", "curand", "lib64"),
                os.path.join(site_dir, "nvidia", "curand", "targets", "x86_64-linux", "lib"),
                os.path.join(site_dir, "nvidia", "cufft", "lib"),
                os.path.join(site_dir, "nvidia", "cufft", "lib64"),
                os.path.join(site_dir, "nvidia", "cufft", "targets", "x86_64-linux", "lib"),
            ])

        existing = env.get("LD_LIBRARY_PATH", "")
        existing_parts = [p for p in existing.split(":") if p]
        merged = []
        for path in candidate_dirs + existing_parts:
            if path and os.path.isdir(path) and path not in merged:
                merged.append(path)

        if merged:
            env["LD_LIBRARY_PATH"] = ":".join(merged)

        return env

    def _ensure_started(self):
        if self._thread is not None and self._thread.is_alive():
            return

        self._shutdown = False
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def _reset_worker(self):
        self.close()
        self._shutdown = False
        self._thread = None

    def _worker_loop(self):
        env_python = self._resolve_env_python_executable()
        worker_env = self._build_worker_env(env_python)
        cmd = [
            env_python,
            self.worker_script_path,
            "--persistent",
        ]

        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=worker_env,
        )

        try:
            while not self._shutdown:
                item = self._request_queue.get()
                if item is None:
                    self._shutdown = True
                    break

                request_payload, response_queue = item
                try:
                    if process.stdin is None or process.stdout is None:
                        raise RuntimeError("Persistent worker process I/O streams are not available")

                    process.stdin.write(json.dumps(request_payload) + "\n")
                    process.stdin.flush()

                    result_line = None
                    recent_logs = []
                    while True:
                        line = process.stdout.readline()
                        if not line:
                            return_code = process.poll()
                            joined_logs = "".join(recent_logs).strip()
                            error_text = "Persistent worker terminated unexpectedly"
                            if return_code is not None:
                                error_text += f" (exit code {return_code})"
                            if joined_logs:
                                error_text += f"\nWorker logs:\n{joined_logs}"
                            raise RuntimeError(error_text)

                        if line:
                            recent_logs.append(line)
                            if len(recent_logs) > 100:
                                recent_logs = recent_logs[-100:]

                        line = line.strip()
                        if line.startswith(self._RESULT_PREFIX):
                            result_line = line[len(self._RESULT_PREFIX):]
                            break
                        if line:
                            print(line)

                    result = json.loads(result_line)
                    response_queue.put(result)
                except Exception as exc:
                    response_queue.put({"ok": False, "error": str(exc)})
        finally:
            try:
                if process.stdin is not None:
                    shutdown_msg = {"shutdown": True}
                    process.stdin.write(json.dumps(shutdown_msg) + "\n")
                    process.stdin.flush()
            except Exception:
                pass

            try:
                process.terminate()
            except Exception:
                pass

            try:
                process.wait(timeout=2)
            except Exception:
                pass

    def run(self, input_data: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
        self._ensure_started()

        temp_dir = os.path.abspath(self.temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
        base_name = self.worker_script_name.replace("_worker.py", "").replace(".py", "")
        request_id = uuid.uuid4().hex
        input_json = os.path.join(temp_dir, f"{base_name}_{request_id}_input.json")
        output_json = os.path.join(temp_dir, f"{base_name}_{request_id}_output.json")

        with open(input_json, "w") as f:
            json.dump(input_data, f)

        response_queue: "queue.Queue[Any]" = queue.Queue(maxsize=1)
        payload = {
            "input_json": input_json,
            "output_json": output_json,
            "verbose": verbose,
        }
        self._request_queue.put((payload, response_queue))
        try:
            result = response_queue.get(timeout=300)
        except queue.Empty:
            self._reset_worker()
            raise RuntimeError("Persistent inference worker timed out waiting for response")

        if not result.get("ok", False):
            error_text = result.get("error", "Persistent inference worker failed")
            lowered = error_text.lower()
            should_retry = (
                "terminated unexpectedly" in lowered
                or "broken pipe" in lowered
                or "i/o streams are not available" in lowered
            )
            if should_retry:
                self._reset_worker()
                self._ensure_started()
                retry_queue: "queue.Queue[Any]" = queue.Queue(maxsize=1)
                self._request_queue.put((payload, retry_queue))
                try:
                    retry_result = retry_queue.get(timeout=300)
                except queue.Empty:
                    self._reset_worker()
                    raise RuntimeError("Persistent inference worker timed out after restart")
                if retry_result.get("ok", False):
                    result = retry_result
                else:
                    result = retry_result

        try:
            if not result.get("ok", False):
                raise RuntimeError(result.get("error", "Persistent inference worker failed"))

            if not os.path.exists(output_json):
                raise RuntimeError(f"Persistent worker did not produce output file: {output_json}")

            with open(output_json, "r") as f:
                return json.load(f)
        finally:
            if os.path.exists(input_json):
                os.remove(input_json)
            if os.path.exists(output_json):
                os.remove(output_json)

    def close(self):
        self._shutdown = True
        self._request_queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=5)
