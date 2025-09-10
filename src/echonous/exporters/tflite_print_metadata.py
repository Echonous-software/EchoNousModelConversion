from __future__ import annotations

import argparse
import json
import os
import sys
import traceback


def _prefer_local_bindings() -> None:
    base_dir = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()
    # Ensure project root is importable first
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)
    # Also insert generated binding dirs if present
    gen_tflite = os.path.join(base_dir, 'tflite')
    gen_tflite_md = os.path.join(base_dir, 'tflite_metadata')
    for p in (gen_tflite, gen_tflite_md):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)


def _import_tflite_bindings():
    errors = []
    for modpath in (
        'tflite.Model',             # generated folder directly on sys.path
    ):
        try:
            return __import__(modpath, fromlist=['Model'])
        except Exception as e:
            errors.append(f"{modpath}: {e}")
    # Last resort: try runtime for reading only
    try:
        return __import__('tflite.Model', fromlist=['Model'])
    except Exception as e:
        errors.append(f"tflite.Model (runtime): {e}")
    raise ImportError("Failed to import tflite Model bindings.\n" + "\n".join(errors))


def _import_metadata_bindings():
    errors = []
    for modpath in (
        'tflite.ModelMetadata',
    ):
        try:
            return __import__(modpath, fromlist=['ModelMetadata'])
        except Exception as e:
            errors.append(f"{modpath}: {e}")
    raise ImportError("Failed to import tflite metadata bindings.\n" + "\n".join(errors))


def _read_file_bytes(path: str) -> bytes:
    with open(path, 'rb') as f:
        return f.read()


def _get_metadata_buffer_index(model, prefer_name: str = "TFLite Metadata") -> int | None:
    # Try to find matching metadata by name
    try:
        mlen = model.MetadataLength()
    except Exception:
        mlen = 0
    if mlen and mlen > 0:
        for i in range(mlen):
            try:
                md = model.Metadata(i)
                nm = md.Name()
                # Convert to str safely
                if isinstance(nm, (bytes, bytearray)):
                    nm = nm.decode('utf-8', errors='ignore')
                if str(nm) == prefer_name:
                    try:
                        return int(md.Buffer())
                    except Exception:
                        continue
            except Exception:
                continue
    return None


def _read_buffer_bytes(model, buf_index: int) -> bytes | None:
    try:
        buf_obj = model.Buffers(buf_index)
    except Exception:
        return None
    # Prefer DataLength/Data(i)
    try:
        if hasattr(buf_obj, 'DataLength') and callable(getattr(buf_obj, 'DataLength')):
            ln = int(buf_obj.DataLength())
            return bytes(bytearray(buf_obj.Data(i) for i in range(ln)))
    except Exception:
        pass
    # Fallback to numpy accessor if present
    try:
        if hasattr(buf_obj, 'DataAsNumpy') and callable(getattr(buf_obj, 'DataAsNumpy')):
            arr = buf_obj.DataAsNumpy()
            return bytes(arr.tobytes())
    except Exception:
        pass
    return None


def _parse_metadata_to_summary(md_buf: bytes) -> dict:
    # Parse with generated metadata bindings
    md_mod = _import_metadata_bindings()
    ModelMetadata = getattr(md_mod, 'ModelMetadata')
    md = ModelMetadata.GetRootAsModelMetadata(md_buf, 0)

    def to_str(x):
        if isinstance(x, (bytes, bytearray)):
            return x.decode('utf-8', errors='ignore')
        return None if x is None else str(x)

    out: dict = {
        "name": to_str(md.Name()),
        "description": to_str(md.Description()),
        "version": to_str(md.Version()),
        "subgraphs": [],
        "raw_length": len(md_buf),
    }
    try:
        s_len = md.SubgraphMetadataLength()
    except Exception:
        s_len = 0
    for i in range(int(s_len)):
        sgm = md.SubgraphMetadata(i)
        nm = to_str(sgm.Name())
        # Inputs
        inputs: list[str] = []
        try:
            ilen = int(sgm.InputTensorMetadataLength())
        except Exception:
            ilen = 0
        for j in range(ilen):
            tm = sgm.InputTensorMetadata(j)
            inputs.append(to_str(tm.Name()) or "")
        # Outputs
        outputs: list[str] = []
        try:
            olen = int(sgm.OutputTensorMetadataLength())
        except Exception:
            olen = 0
        for j in range(olen):
            tm = sgm.OutputTensorMetadata(j)
            outputs.append(to_str(tm.Name()) or "")
        out["subgraphs"].append({
            "name": nm,
            "inputs": inputs,
            "outputs": outputs,
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Print embedded TFLite flatbuffer metadata")
    parser.add_argument("tflite_path", help="Path to .tflite file")
    args = parser.parse_args()

    _prefer_local_bindings()

    try:
        tflite_bytes = _read_file_bytes(args.tflite_path)
    except Exception as e:
        print(f"Failed to read file: {e}", file=sys.stderr)
        return 2

    try:
        mod_model = _import_tflite_bindings()
        Model = getattr(mod_model, 'Model')
    except Exception as e:
        print(f"Failed to import TFLite bindings: {e}", file=sys.stderr)
        return 2

    try:
        # Match packer style: use GetRootAs with bytes + 0 offset if available; otherwise fallback
        # Many codegens expose GetRootAsModel; handle both.
        if hasattr(Model, 'GetRootAs') and callable(getattr(Model, 'GetRootAs')):
            model = Model.GetRootAs(tflite_bytes, 0)
        elif hasattr(Model, 'GetRootAsModel') and callable(getattr(Model, 'GetRootAsModel')):
            model = Model.GetRootAsModel(tflite_bytes, 0)
        else:
            # As a last resort: construct table via tflite-runtime style
            model = Model.GetRootAsModel(tflite_bytes, 0)  # type: ignore[attr-defined]
    except Exception as e:
        print(f"Failed to parse TFLite model: {e}", file=sys.stderr)
        return 2

    # Try to resolve buffer index via metadata entries
    buf_index = _get_metadata_buffer_index(model, prefer_name="TFLite Metadata")
    resolved_via_name = buf_index is not None
    if buf_index is None:
        buf_index = 81

    # Read buffer
    md_buf = _read_buffer_bytes(model, int(buf_index))
    if not md_buf:
        src = "resolved name" if resolved_via_name else "fallback index 81"
        print(f"Failed to read metadata buffer ({src}).", file=sys.stderr)
        return 1

    # Parse and print summary
    try:
        summary = _parse_metadata_to_summary(md_buf)
    except Exception as e:
        print("Failed to parse metadata buffer:", file=sys.stderr)
        print(str(e), file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return 1

    print(json.dumps({
        "resolved_via_name": resolved_via_name,
        "buffer_index": int(buf_index),
        "metadata": summary,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


