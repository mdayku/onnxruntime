# Model Inspect Scaffolds

This canvas contains initial scaffolding for the following files:

- `python/tools/model_inspect.py`
- `python/tools/model_inspect_compare.py`
- `core/graph/model_inspector.h`
- `core/graph/model_inspector.cc`

You can download this canvas and split these into separate files in your repo.

---

## python/tools/model_inspect.py

```python
#!/usr/bin/env python
"""CLI for inspecting model graphs.

This is intentionally framework-agnostic. Plug in your actual loader /
introspection logic inside `core.graph.model_inspector.ModelInspector`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

try:
    # You can implement this as a pure-Python module or expose a C++ core
    # via pybind11. The interface expected here is defined in
    # `core/graph/model_inspector.h`.
    from core.graph.model_inspector import ModelInspector  # type: ignore
except Exception:  # pragma: no cover - stub fallback

    class ModelInspector:  # type: ignore[override]
        """Very small stub so the CLI still runs.

        Replace with the real implementation.
        """

        def __init__(self, path: str) -> None:
            self.path = str(path)

        @classmethod
        def from_file(cls, path: str) -> "ModelInspector":
            return cls(path)

        def summary(self) -> Dict[str, Any]:
            return {
                "path": self.path,
                "num_nodes": 0,
                "num_parameters": 0,
                "input_shapes": [],
                "output_shapes": [],
            }

        def to_json(self) -> Dict[str, Any]:
            return self.summary()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="model_inspect",
        description=(
            "Inspect a model graph and print high-level statistics, "
            "I/O shapes, and an optional JSON dump."
        ),
    )

    parser.add_argument(
        "model",
        type=str,
        help="Path to the model file to inspect (e.g., .onnx, .pt, etc.)",
    )

    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a human-readable summary to stdout.",
    )

    parser.add_argument(
        "--json",
        metavar="PATH",
        type=str,
        help="Optional path to write a JSON dump of the model graph stats.",
    )

    parser.add_argument(
        "--pretty-json",
        action="store_true",
        help="Pretty-print the JSON summary to stdout as well.",
    )

    return parser


def _print_summary(summary: Dict[str, Any]) -> None:
    print("Model summary:")
    print(f"  Path: {summary.get('path')}")
    print(f"  Num nodes: {summary.get('num_nodes')}")
    print(f"  Num parameters: {summary.get('num_parameters')}")

    inputs = summary.get("input_shapes") or []
    outputs = summary.get("output_shapes") or []

    if inputs:
        print("  Inputs:")
        for i, shape in enumerate(inputs):
            print(f"    [{i}] shape={shape}")

    if outputs:
        print("  Outputs:")
        for i, shape in enumerate(outputs):
            print(f"    [{i}] shape={shape}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    model_path = Path(args.model)
    if not model_path.is_file():
        parser.error(f"Model path does not exist or is not a file: {model_path}")

    inspector = ModelInspector.from_file(str(model_path))
    summary = inspector.summary()

    if args.summary:
        _print_summary(summary)

    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(inspector.to_json(), f, indent=2)
        print(f"JSON summary written to {json_path}")

    if args.pretty_json and not args.json:
        # Pretty-print to stdout only
        print(json.dumps(inspector.to_json(), indent=2))
    elif args.pretty_json and args.json:
        # Print the same content we wrote
        with Path(args.json).open("r", encoding="utf-8") as f:
            data = json.load(f)
        print(json.dumps(data, indent=2))

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
```

---

## python/tools/model_inspect_compare.py

```python
#!/usr/bin/env python
"""CLI for comparing two model graphs.

This builds on the same `ModelInspector` interface used in `model_inspect.py`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from core.graph.model_inspector import ModelInspector  # type: ignore
except Exception:  # pragma: no cover - stub fallback

    class ModelInspector:  # type: ignore[override]
        def __init__(self, path: str) -> None:
            self.path = path

        @classmethod
        def from_file(cls, path: str) -> "ModelInspector":
            return cls(path)

        def summary(self) -> Dict[str, Any]:
            return {"path": self.path}


@dataclass
class CompareResult:
    left_path: str
    right_path: str
    are_compatible: bool
    details: Dict[str, Any]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="model_inspect_compare",
        description="Compare two model graphs and highlight key differences.",
    )

    parser.add_argument("left", type=str, help="Path to the first (reference) model.")
    parser.add_argument("right", type=str, help="Path to the second (candidate) model.")

    parser.add_argument(
        "--json",
        metavar="PATH",
        help="Optional path to write the comparison report as JSON.",
    )

    parser.add_argument(
        "--strict-io",
        action="store_true",
        help="Require identical input and output shapes to mark as compatible.",
    )

    return parser


def compare_models(
    left: ModelInspector,
    right: ModelInspector,
    *,
    strict_io: bool = False,
) -> CompareResult:
    left_summary = left.summary()
    right_summary = right.summary()

    details: Dict[str, Any] = {
        "left": left_summary,
        "right": right_summary,
    }

    compatible = True

    # Example comparison heuristic; tweak as needed
    if strict_io:
        compatible = (
            left_summary.get("input_shapes") == right_summary.get("input_shapes")
            and left_summary.get("output_shapes") == right_summary.get("output_shapes")
        )

    details["metrics"] = {
        "node_count_delta": (
            (right_summary.get("num_nodes") or 0)
            - (left_summary.get("num_nodes") or 0)
        ),
        "param_count_delta": (
            (right_summary.get("num_parameters") or 0)
            - (left_summary.get("num_parameters") or 0)
        ),
    }

    return CompareResult(
        left_path=str(left_summary.get("path")),
        right_path=str(right_summary.get("path")),
        are_compatible=compatible,
        details=details,
    )


def _print_human(result: CompareResult) -> None:
    print("Model comparison:")
    print(f"  Left:  {result.left_path}")
    print(f"  Right: {result.right_path}")
    print(f"  Compatible: {'yes' if result.are_compatible else 'no'}")

    metrics = result.details.get("metrics", {})
    print("  Metrics:")
    print(f"    Δ nodes: {metrics.get('node_count_delta')}")
    print(f"    Δ params: {metrics.get('param_count_delta')}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    left_path = Path(args.left)
    right_path = Path(args.right)

    if not left_path.is_file():
        parser.error(f"Left model does not exist: {left_path}")
    if not right_path.is_file():
        parser.error(f"Right model does not exist: {right_path}")

    left = ModelInspector.from_file(str(left_path))
    right = ModelInspector.from_file(str(right_path))

    result = compare_models(left, right, strict_io=bool(args.strict_io))

    _print_human(result)

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "left_path": result.left_path,
                    "right_path": result.right_path,
                    "are_compatible": result.are_compatible,
                    "details": result.details,
                },
                f,
                indent=2,
            )
        print(f"JSON comparison written to {out}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
```

---

## core/graph/model_inspector.h

```cpp
#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>

namespace core {
namespace graph {

struct TensorShape {
  std::vector<int64_t> dims;
};

struct ModelSummary {
  std::string path;
  std::size_t num_nodes = 0;
  std::size_t num_parameters = 0;
  std::vector<TensorShape> input_shapes;
  std::vector<TensorShape> output_shapes;
};

// Optional C++ core for fast inspection. You can expose this via pybind11
// to be used by the Python CLIs.
class ModelInspector {
 public:
  explicit ModelInspector(std::string path);

  // Factory to construct from file; can perform the heavy parsing in the
  // implementation.
  static ModelInspector FromFile(const std::string& path);

  // Return a high-level summary for logging / comparison.
  ModelSummary Summary() const;

  // Serialize the summary (and any additional fields you decide to add)
  // into a JSON-like structure. To keep this header-only, we simply
  // expose a string here; implementation can pick your JSON library.
  std::string ToJsonString(bool pretty = true) const;

 private:
  std::string path_;
  ModelSummary summary_;

  // TODO: add actual graph representation, e.g. nodes, edges, attributes.
};

}  // namespace graph
}  // namespace core
```

---

## core/graph/model_inspector.cc

```cpp
#include "core/graph/model_inspector.h"

#include <sstream>

namespace core {
namespace graph {

ModelInspector::ModelInspector(std::string path)
    : path_(std::move(path)) {
  // TODO: Parse the model file at `path_` and populate `summary_`.
  // For now, we just store the path.
  summary_.path = path_;
}

ModelInspector ModelInspector::FromFile(const std::string& path) {
  return ModelInspector(path);
}

ModelSummary ModelInspector::Summary() const {
  return summary_;
}

std::string ModelInspector::ToJsonString(bool /*pretty*/) const {
  // Minimal hand-rolled JSON; replace with nlohmann::json or similar.
  std::ostringstream oss;
  oss << "{";
  oss << "\"path\":\"" << summary_.path << "\",";
  oss << "\"num_nodes\":" << summary_.num_nodes << ",";
  oss << "\"num_parameters\":" << summary_.num_parameters;
  // You can extend this to include shapes if you want.
  oss << "}";
  return oss.str();
}

}  // namespace graph
}  // namespace core
```

