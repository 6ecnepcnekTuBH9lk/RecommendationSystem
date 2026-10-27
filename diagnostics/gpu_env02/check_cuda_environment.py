"""Verify the existing venv in a parent and the same spawn context as the runner."""
import json
import multiprocessing as mp
from pathlib import Path
import sys

import torch


def inspect_cuda(smoke=False):
    result = {
        "sys_executable": sys.executable, "sys_version": sys.version,
        "sys_prefix": sys.prefix, "torch_file": torch.__file__,
        "torch_version": torch.__version__, "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_is_built": torch.backends.cuda.is_built(),
    }
    if result["cuda_available"]:
        properties = torch.cuda.get_device_properties(0)
        result.update(gpu_name=properties.name,
                      capability=list(torch.cuda.get_device_capability(0)),
                      total_vram_bytes=properties.total_memory)
        if smoke:
            tensor = torch.randn(32, 32, device="cuda", requires_grad=True)
            product = tensor @ tensor.t()
            product.square().mean().backward()
            torch.cuda.synchronize()
            cpu_result = product.detach().cpu()
            assert cpu_result.numpy().shape == (32, 32)
            assert torch.isfinite(cpu_result).all().item()
            assert tensor.grad is not None and torch.isfinite(tensor.grad).all().item()
            result["smoke_test"] = {
                "passed": True, "operation": "matrix multiply, squared mean, backward",
                "returned_device": str(cpu_result.device),
                "shape": list(cpu_result.shape), "finite_output_and_gradient": True,
                "numpy_interop_passed": True,
            }
    return result


def worker(connection):
    try:
        connection.send({"report": inspect_cuda()})
    except Exception as exc:
        connection.send({"error_type": type(exc).__name__, "error": str(exc)})
    finally:
        connection.close()


def main():
    root = Path(__file__).resolve().parent
    report = {"status": "checking", "parent": inspect_cuda(smoke=True)}
    if not report["parent"]["cuda_available"]:
        report["status"] = "cuda_unavailable"
        (root / "cuda_environment.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 1
    context = mp.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=worker, args=(sender,))
    process.start()
    sender.close()
    try:
        if not receiver.poll(90):
            raise RuntimeError("Diagnostic child did not return within 90 seconds")
        response = receiver.recv()
        process.join(30)
        if "report" not in response:
            raise RuntimeError(str(response))
        report["worker"] = response["report"]
        keys = ("sys_executable", "sys_prefix", "torch_version", "torch_cuda_build", "gpu_name")
        assert report["worker"]["cuda_available"]
        assert all(report["parent"][key] == report["worker"][key] for key in keys)
        assert process.exitcode == 0
        assert report["parent"]["torch_version"] == "2.5.1+cu124"
        report.update(status="passed", multiprocessing_context="spawn",
                      parent_worker_environment_matches=True)
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
        receiver.close()
    (root / "cuda_environment.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

