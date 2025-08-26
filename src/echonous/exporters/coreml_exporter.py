"""
CoreML export API for converting a PyTorch nn.Module (via Model dataclass) to CoreML.

Supports coremltools >= 4 and Python >= 3.8. Provides:
- Importable API (no CLI)
- Precision/compute configuration
- Input/output naming from Model dataclass
- Image input handling (grayscale, scale)
- Validation by comparing PyTorch and CoreML outputs on random inputs
- Metadata embedding and sidecar JSON
- Flexible shape branching stub
- Custom op registry with basic composition detection

Notes on CPU vs GPU for tracing/inputs:
- We run model.eval() on CPU and create torch.rand inputs on CPU to avoid device
  mismatches and to minimize numerical divergence due to different GPU kernels.
  CoreML conversion expects CPU tensors in most scenarios; keeping CPU ensures
  consistent tracing behavior across environments.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import json
import time
import os

import torch
import torch.nn as nn
import numpy as np

from echonous.models import Model, ModuleInputOutput


class CoreMLExportError(Exception):
    pass


class CoreMLExportAPI:
    """
    CoreML export API centered around the `Model` dataclass.

    Usage:
        api = CoreMLExportAPI()
        api.handleCustomOp("my_custom_op", my_handler)
        metadata = api.convert_model_dataclass(model, 
                                               output_path="outputs/output.mlmodel",
                                               precision="Float32",
                                               compute_units="ALL",
                                               tolerance=1e-3,
                                               flexible_shapes=False)
    """

    def __init__(self) -> None:
        self.custom_op_handlers: Dict[str, Callable[[str], Optional[nn.Module]]] = {}

    def handleCustomOp(self, name: str, handler: Callable[[str], Optional[nn.Module]]) -> None:
        """
        Register a custom op handler.

        Args:
            name: Operation name (e.g., from TorchScript graph like 'prim::PythonOp/YourOp')
            handler: Function that accepts the op name and may return an alternative
                     nn.Module with a supported implementation, or None if not handled.
        """
        if not isinstance(name, str) or not name:
            raise ValueError("Custom op name must be a non-empty string")
        if not callable(handler):
            raise ValueError("Custom op handler must be callable")
        self.custom_op_handlers[name] = handler

    def convert_model_dataclass(
        self,
        model_dc: Model,
        output_path: str,
        precision: str = "Float32",  # 'Float32' | 'Float16'
        compute_units: str = "ALL",   # 'ALL' | 'CPU_ONLY' | 'CPU_AND_GPU' | 'CPU_AND_NE'
        tolerance: float = 1e-3,
        flexible_shapes: bool = False,
    ) -> Dict[str, Any]:
        """
        Convert the PyTorch model from the `Model` dataclass to CoreML and save.

        Args:
            model_dc: Model dataclass instance with module, inputs, outputs, metadata
            output_path: where to save the .mlmodel
            precision: 'Float32' or 'Float16'
            compute_units: preferred compute target (best effort depending on coremltools version)
            tolerance: absolute tolerance for validation comparing PyTorch vs CoreML outputs
            flexible_shapes: when True, branch to flexible-shape path (stub -> NotImplemented)

        Returns:
            Metadata dictionary about the conversion and validation.
        """
        start_time = time.time()

        if flexible_shapes:
            self._handle_flexible_shapes_stub()

        module = model_dc.module
        if not isinstance(module, nn.Module):
            raise CoreMLExportError("model_dc.module must be an instance of torch.nn.Module")

        # Ensure eval mode on CPU for consistent tracing and inputs
        module = module.eval()

        # Build example inputs for tracing and validation
        torch_inputs = self._build_torch_inputs(model_dc.inputs)

        # Early graph scan for custom ops; attempt composition detection or handler
        #self._preflight_custom_ops(module, torch_inputs)

        # Perform conversion
        coreml_model, ct_version, spec_version = self._convert(module, model_dc, torch_inputs, precision, compute_units)

        # Rename outputs to match dataclass outputs if necessary
        self._apply_output_names(coreml_model, model_dc.outputs)

        # Embed metadata
        self._embed_metadata(coreml_model, model_dc)

        # Save model
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        coreml_model.save(output_path)

        # Validate by comparing outputs
        validation = self._validate(module, coreml_model, model_dc, torch_inputs, tolerance)

        # Sidecar metadata JSON
        sidecar_path = self._write_sidecar_metadata(output_path, model_dc, precision, compute_units, tolerance, ct_version, spec_version, validation)

        return {
            "output_model_path": output_path,
            "sidecar_metadata_path": sidecar_path,
            "coremltools_version": ct_version,
            "coreml_spec_version": spec_version,
            "precision": precision,
            "compute_units": compute_units,
            "validation": validation,
            "elapsed_seconds": round(time.time() - start_time, 3),
        }

    def _build_torch_inputs(self, inputs: List[ModuleInputOutput]) -> List[torch.Tensor]:
        tensors: List[torch.Tensor] = []
        for io in inputs:
            shape = tuple(int(s) for s in io.shape)
            if any(dim <= 0 for dim in shape):
                raise CoreMLExportError(f"Input '{io.name}' has non-positive dimension in shape: {shape}")
            if io.type == 'image':
                # image type must respect scale:
                # coreml model will take 0-255 * scale as input, so apply the same logic here
                tensor = torch.rand(shape, dtype=torch.float32) * 255.0 * io.scale
            else:
                tensor = torch.rand(shape, dtype=torch.float32)
            tensors.append(tensor)
        return tensors

    def _preflight_custom_ops(self, module: nn.Module, example_inputs: List[torch.Tensor]) -> None:
        """
        Trace the model graph and look for ops that may be problematic.
        If custom ops are discovered, try handler; otherwise check if they're just
        compositions of supported ops and allow if so.
        """
        with torch.no_grad():
            try:
                print("### TRACING ###")
                traced = torch.jit.trace(module, example_inputs if len(example_inputs) > 1 else example_inputs[0])
                graph_str = str(traced.inlined_graph) if hasattr(traced, "inlined_graph") else str(traced.graph)
            except Exception as e:
                # If tracing fails here, defer to converter which will surface a clear error
                return

        unknown_ops = self._find_unknown_ops_in_graph(graph_str)
        if not unknown_ops:
            print("### NO UNKNOWN OPS ###")
            return

        # Attempt to handle or validate composition
        unresolved: List[str] = []
        for op in unknown_ops:
            if op in self.custom_op_handlers:
                try:
                    replacement = self.custom_op_handlers[op](op)
                    if isinstance(replacement, nn.Module):
                        # User provided a replacement module; swap and re-trace for future stages
                        module.__class__ = replacement.__class__  # type: ignore[attr-defined]
                        module.__dict__ = replacement.__dict__
                    # If None, assume handler performed necessary side effects
                except Exception:
                    unresolved.append(op)
                continue

            # If op looks like a PythonOp, try to see if the containing submodule is composite
            if "prim::PythonOp" in op:
                if self._looks_like_composition(module, example_inputs):
                    continue
                unresolved.append(op)
            else:
                # If not recognized namespace (e.g., custom::), mark unresolved
                if not op.startswith("aten::") and not op.startswith("prim::"):
                    if self._looks_like_composition(module, example_inputs):
                        continue
                    unresolved.append(op)

        if unresolved:
            raise CoreMLExportError(
                f"Encountered unsupported custom ops without handlers: {sorted(set(unresolved))}. "
                f"Register handlers via handleCustomOp(name, fn) or refactor to supported ops."
            )

    def _find_unknown_ops_in_graph(self, graph_str: str) -> List[str]:
        ops: List[str] = []
        for line in graph_str.splitlines():
            line = line.strip()
            # Lines often look like: %xy = aten::conv2d(...), or %z = prim::PythonOp[name="..."](…)
            if " = " in line and "::" in line:
                try:
                    rhs = line.split(" = ", 1)[1]
                    op = rhs.split("(", 1)[0]
                    ops.append(op)
                except Exception:
                    continue
        # Filter to potentially problematic namespaces
        return sorted(set([op for op in ops if not op.startswith("aten::")]))

    def _looks_like_composition(self, module: nn.Module, example_inputs: List[torch.Tensor]) -> bool:
        """
        Heuristic: if we can trace the module and the resulting graph contains only aten:: ops
        (no prim::PythonOp or custom::), we treat it as a composition of supported ops.
        """
        try:
            with torch.no_grad():
                traced = torch.jit.trace(module, example_inputs if len(example_inputs) > 1 else example_inputs[0])
                graph_str = str(traced.inlined_graph) if hasattr(traced, "inlined_graph") else str(traced.graph)
            return ("prim::PythonOp" not in graph_str) and all(
                ("::" not in line) or ("aten::" in line) or ("prim::" in line)
                for line in graph_str.splitlines()
            )
        except Exception:
            return False

    def _convert(
        self,
        module: nn.Module,
        model_dc: Model,
        example_inputs: List[torch.Tensor],
        precision: str,
        compute_units: str,
        storage: str="Float32",
    ) -> Tuple[Any, str, Optional[str]]:
        # Local import to allow environments without coremltools to import the module
        try:
            import coremltools as ct  # type: ignore
        except Exception as e:
            raise CoreMLExportError(f"coremltools is required for conversion: {e}")

        ct_version = getattr(ct, "__version__", "unknown")
        major = 0
        try:
            major = int(str(ct_version).split(".")[0])
        except Exception:
            pass

        # Prepare CoreML input types using names and grayscale scale
        coreml_inputs: List[Any] = []
        for io in model_dc.inputs:
            shape = tuple(int(s) for s in io.shape)
            if io.type == 'image' and len(shape) == 4:
                image_kwargs: Dict[str, Any] = {
                    "name": io.name,
                    "shape": shape,
                }
                # Scale is passed directly
                if hasattr(ct, "ImageType"):
                    # color layout handling across versions
                    try:
                        image_kwargs["scale"] = float(io.scale)
                    except Exception:
                        image_kwargs["scale"] = 1.0
                    try:
                        # Prefer explicit grayscale layout when available
                        image_kwargs["color_layout"] = getattr(ct.colorlayout, "GRAYSCALE")
                    except Exception:
                        pass
                    coreml_inputs.append(ct.ImageType(**image_kwargs))
                else:
                    raise CoreMLExportError("coremltools.ImageType not available")
            else:
                # Generic tensor input
                if hasattr(ct, "TensorType"):
                    coreml_inputs.append(ct.TensorType(name=io.name, shape=shape))
                else:
                    raise CoreMLExportError("coremltools.TensorType not available")

        # Example inputs for tracing
        ex_inputs = example_inputs if len(example_inputs) > 1 else example_inputs[0]

        # Trace model
        with torch.no_grad():
            traced = torch.jit.trace(module, ex_inputs)

        # Build convert args with version-aware options
        convert_args: Dict[str, Any] = {
            "model": traced,
            "inputs": coreml_inputs,
        }

        # Precision mapping
        try:
            if precision == "Float16":
                # Version-aware precision handling
                if hasattr(ct, "precision") and hasattr(ct.precision, "FLOAT16"):
                    convert_args["compute_precision"] = ct.precision.FLOAT16
                else:
                    convert_args["compute_precision"] = "float16"  # best-effort for older versions
            else:
                if hasattr(ct, "precision") and hasattr(ct.precision, "FLOAT32"):
                    convert_args["compute_precision"] = ct.precision.FLOAT32
                else:
                    convert_args["compute_precision"] = "float32"
        except Exception:
            # Omit compute_precision if unsupported by current ct
            convert_args.pop("compute_precision", None)
        # Storage

        # Compute units (best-effort)
        try:
            if hasattr(ct, "ComputeUnit"):
                cu_map = {
                    "ALL": ct.ComputeUnit.ALL,
                    "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
                    "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
                }
                # Include Neural Engine if present
                if hasattr(ct.ComputeUnit, "CPU_AND_NE"):
                    cu_map["CPU_AND_NE"] = ct.ComputeUnit.CPU_AND_NE
                convert_args["compute_units"] = cu_map.get(compute_units, ct.ComputeUnit.ALL)
        except Exception:
            convert_args.pop("compute_units", None)

        # Minimum deployment target selection (iOS/macOS agnostic default)
        try:
            target_map = {
                4: getattr(ct.target, "iOS13", None),
                5: getattr(ct.target, "iOS14", None),
                6: getattr(ct.target, "iOS15", None),
            }
            target = target_map.get(major)
            if target is not None:
                convert_args["minimum_deployment_target"] = target
        except Exception:
            pass

        # Perform conversion
        try:
            coreml_model = ct.convert(**convert_args)
        except Exception as e:
            # Retry without precision/compute hints
            convert_args.pop("compute_precision", None)
            convert_args.pop("compute_units", None)
            try:
                coreml_model = ct.convert(**convert_args)
            except Exception as e2:
                raise CoreMLExportError(f"CoreML conversion failed: {e2}")

        # Extract spec version if possible
        spec_version: Optional[str]
        try:
            spec = coreml_model.get_spec()
            spec_version = str(getattr(spec, "specificationVersion", "unknown"))
        except Exception:
            spec_version = None

        return coreml_model, str(ct_version), spec_version

    def _apply_output_names(self, coreml_model: Any, outputs: List[ModuleInputOutput]) -> None:
        try:
            import coremltools as ct  # type: ignore
        except Exception:
            return
        try:
            spec = coreml_model.get_spec()
            existing = list(spec.description.output)
            rename_map: List[Tuple[str, str]] = []
            for idx, out_desc in enumerate(existing):
                if idx < len(outputs):
                    old_name = out_desc.name
                    new_name = outputs[idx].name
                    if old_name != new_name:
                        rename_map.append((old_name, new_name))
            for old, new in rename_map:
                try:
                    coreml_model = ct.utils.rename_feature(coreml_model, old, new)
                except Exception:
                    # If rename fails, continue; validation will still refer to CoreML names
                    pass
        except Exception:
            pass

    def _embed_metadata(self, coreml_model: Any, model_dc: Model) -> None:
        try:
            spec = coreml_model.get_spec()
        except Exception:
            return
        # High-level fields
        try:
            coreml_model.short_description = model_dc.description or ""
        except Exception:
            pass
        # User-defined metadata
        try:
            md = {
                "name": model_dc.name,
                "version": model_dc.version,
                "description": model_dc.description,
            }
            # Best-effort storage depending on ct version
            if hasattr(coreml_model, "user_defined_metadata") and hasattr(coreml_model.user_defined_metadata, "update"):
                coreml_model.user_defined_metadata.update({k: str(v) for k, v in md.items()})
        except Exception:
            pass

    def _validate(
        self,
        module: nn.Module,
        coreml_model: Any,
        model_dc: Model,
        torch_inputs: List[torch.Tensor],
        tolerance: float,
    ) -> Dict[str, Any]:
        # PyTorch forward
        print("### VALIDATING ###")
        with torch.no_grad():
            pt_out = module(*torch_inputs) if len(torch_inputs) > 1 else module(torch_inputs[0])

        # Normalize to list of np arrays
        pt_outputs: List[np.ndarray] = []
        if isinstance(pt_out, (list, tuple)):
            for o in pt_out:
                pt_outputs.append(self._to_numpy(o))
        else:
            pt_outputs.append(self._to_numpy(pt_out))

        # CoreML prediction
        # Build input dict: pass PIL Image for image-typed inputs, numpy for tensors
        input_dict: Dict[str, Any] = {}
        for idx, io in enumerate(model_dc.inputs):
            tensor = torch_inputs[idx].detach().cpu()
            if io.type == 'image':
                try:
                    input_dict[io.name] = self._tensor_to_pil_image(tensor, io.scale)
                except Exception:
                    print("### FALLBACK ###")
                    # Fallback to numpy array (uint8 HWC) if PIL unavailable
                    arr = tensor.numpy()
                    if arr.ndim == 4:
                        arr = arr[0]
                    if arr.ndim == 3 and arr.shape[0] in (1, 3):
                        arr = np.transpose(arr, (1, 2, 0))
                    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
                    if arr.ndim == 3 and arr.shape[2] == 1:
                        arr = arr[:, :, 0]
                    input_dict[io.name] = arr
            else:
                input_dict[io.name] = tensor.numpy().astype(np.float32)

        try:
            ml_out = coreml_model.predict(input_dict)
        except Exception as e:
            return {
                "passed": False,
                "error": f"CoreML predict failed: {e}",
            }

        # Extract CoreML outputs in provided order if possible
        cm_arrays: List[np.ndarray] = []
        if model_dc.outputs:
            for out in model_dc.outputs:
                val = ml_out.get(out.name)
                if val is None and len(cm_arrays) < len(ml_out):
                    # Fallback: take next item
                    val = list(ml_out.values())[len(cm_arrays)]
                cm_arrays.append(self._to_numpy(val))
        else:
            for v in ml_out.values():
                cm_arrays.append(self._to_numpy(v))

        # Compare elementwise with absolute tolerance
        comparisons: List[Dict[str, Any]] = []
        passed = True
        for i in range(min(len(pt_outputs), len(cm_arrays))):
            a = pt_outputs[i]
            b = cm_arrays[i]
            same_shape = a.shape == b.shape
            if not same_shape:
                print("### INCORRECT SHAPE ###")
                passed = False
                diff = float("inf")
            else:
                #tester = nn.MSELoss()
                #mse_diff = tester(pt_outputs, cm_arrays)
                #passed = np.abs(mse_diff) < tolerance
                diff = float(np.max(np.abs(a - b))) if a.size and b.size else 0.0
                if not np.allclose(a, b, atol=tolerance, rtol=0):
                    passed = False
            comparisons.append({
                "index": i,
                "pytorch_shape": list(a.shape),
                "coreml_shape": list(b.shape),
                "pytorch_value": a[0,0].astype(float),
                "coreml_value": b[0,0].astype(float),
                "max_abs_diff": None if not same_shape else round(diff, 6),
                "within_tolerance": same_shape and (diff <= tolerance),
            })

        return {
            "passed": passed,
            "tolerance": tolerance,
            "comparisons": comparisons,
        }

    def _tensor_to_pil_image(self, tensor: torch.Tensor, scale: float) -> Any:
        """
        Convert a float tensor in [0,255*scale] with shape NCHW/CHW/HWC to a PIL Image.
        Grayscale -> 'L', RGB -> 'RGB'. Removes batch dim if present.
        """
        try:
            from PIL import Image  # type: ignore
        except Exception as e:
            raise CoreMLExportError(f"PIL is required for image input validation: {e}")

        arr = tensor.detach().cpu().numpy()
        # Remove batch dim if present
        if arr.ndim == 4:
            arr = arr[0]

        mode = None
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            # CHW -> HWC
            c = arr.shape[0]
            arr = np.transpose(arr, (1, 2, 0))
            mode = 'L' if c == 1 else 'RGB'
        elif arr.ndim == 2:
            mode = 'L'
        elif arr.ndim == 3 and arr.shape[2] in (1, 3):
            # Already HWC
            c = arr.shape[2]
            mode = 'L' if c == 1 else 'RGB'
        else:
            # Attempt to coerce to 2D grayscale
            if arr.ndim >= 2:
                arr = arr.reshape(arr.shape[-2], arr.shape[-1])
                mode = 'L'
            else:
                raise CoreMLExportError(f"Unsupported image tensor shape for PIL conversion: {tensor.shape}")

        arr8 = (arr / scale).clip(0, 255).astype(np.uint8)
        if mode == 'L' and arr8.ndim == 3 and arr8.shape[2] == 1:
            print("### FORCING 1CH ###")
            arr8 = arr8[:, :, 0]
            arr = arr[:, :, 0]
        return Image.fromarray(arr8.astype(np.uint8), mode=mode)

    def _to_numpy(self, x: Any) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        if isinstance(x, np.ndarray):
            return x
        # CoreML might return types convertible to np.array
        return np.array(x)

    def _write_sidecar_metadata(
        self,
        output_path: str,
        model_dc: Model,
        precision: str,
        compute_units: str,
        tolerance: float,
        ct_version: str,
        spec_version: Optional[str],
        validation: Dict[str, Any],
    ) -> str:
        meta = {
            "name": model_dc.name,
            "version": model_dc.version,
            "description": model_dc.description,
            "inputs": [asdict(i) for i in model_dc.inputs],
            "outputs": [asdict(o) for o in model_dc.outputs],
            "precision": precision,
            "compute_units": compute_units,
            "tolerance": tolerance,
            "coremltools_version": ct_version,
            "coreml_specification_version": spec_version,
            "validation": validation,
            "generated_at": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        }
        sidecar = output_path if output_path.endswith('.mlpackage') else f"{output_path}.mlpackage"
        sidecar = sidecar.replace('.mlpackage', '_metadata.json')
        try:
            with open(sidecar, 'w') as f:
                json.dump(meta, f, indent=2)
        except Exception as e:
            raise CoreMLExportError(f"Failed to write sidecar metadata: {e}")
        return sidecar

    def _handle_flexible_shapes_stub(self) -> None:
        raise NotImplementedError(
            "Flexible input shape ranges are not implemented yet. "
            "Stub present for future implementation."
        )


def convert_model_to_coreml(
    model_dc: Model,
    output_path: str,
    precision: str = "Float32",
    compute_units: str = "ALL",
    tolerance: float = 1e-3,
    flexible_shapes: bool = False,
) -> Dict[str, Any]:
    """
    Convenience function wrapping CoreMLExportAPI.convert_model_dataclass.
    """
    api = CoreMLExportAPI()
    return api.convert_model_dataclass(
        model_dc,
        output_path=output_path,
        precision=precision,
        compute_units=compute_units,
        tolerance=tolerance,
        flexible_shapes=flexible_shapes,
    )

def convert_all_models(output_dir: Path):
    from echonous.models.loaders import load_all_models
    models = load_all_models()
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        output_path = output_dir / f"{name.replace('.', '_')}.mlpackage"
        convert_model_to_coreml(model, str(output_path))

if __name__ == '__main__':
    convert_all_models(Path(__file__).parent.parent.parent.parent / "export" / "coreml")
