import json
import sys
from dataclasses import dataclass
from typing import List, Optional, Union, Any, Dict
from pathlib import Path


@dataclass
class InstancePathElement:
    """Represents a single element in an instance path."""
    instance_name: str
    module_name: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InstancePathElement':
        return cls(
            instance_name=data["instance_name"],
            module_name=data["module_name"]
        )


@dataclass
class Object:
    """Represents an object in the dataflow graph."""
    instance_path: List[InstancePathElement]
    name: str
    bit_pos: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Object':
        instance_path = [
            InstancePathElement.from_dict(elem) 
            for elem in data["instance_path"]
        ]
        return cls(
            instance_path=instance_path,
            name=data["name"],
            bit_pos=data["bit_pos"]
        )


@dataclass
class DebugPoint:
    """Represents a debug point in the path history."""
    object: Object
    delay: int
    comment: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DebugPoint':
        return cls(
            object=Object.from_dict(data["object"]),
            delay=data["delay"],
            comment=data["comment"]
        )


@dataclass
class OpenPath:
    """Represents an open path with fan-in, delay, and history."""
    fan_in: Object
    delay: int
    history: List[DebugPoint]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OpenPath':
        history = [
            DebugPoint.from_dict(point) 
            for point in data["history"]
        ]
        return cls(
            fan_in=Object.from_dict(data["fan_in"]),
            delay=data["delay"],
            history=history
        )


@dataclass
class DataflowPath:
    """Represents a complete dataflow path from fan-out to fan-in."""
    fan_out: Object  # Could be Object or output port, but JSON shows Object structure
    path: OpenPath
    root: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataflowPath':
        return cls(
            fan_out=Object.from_dict(data["fan_out"]),
            path=OpenPath.from_dict(data["path"]),
            root=data["root"]
        )

    def to_flamegraph_format_detailed(self) -> str:
        """
        Convert this path to detailed FlameGraph format showing the full timing path.

        This version includes both fan-in and fan-out information in the stack.

        Args:
            include_history: If True, include debug history points in the stack

        Returns:
            String in FlameGraph format with detailed path information
        """
        stack_elements = []
        trace = []

        # # Start with root module
        # stack_elements.append(f"root:{self.root}")

        # # Add fan-in hierarchy (input side of the path)
        fan_in_hierarchy = self._build_hierarchy_string(self.path.fan_in, "fan_in")
        prev = fan_in_hierarchy
        prev_delay = 0
        if fan_in_hierarchy:
            stack_elements.append(fan_in_hierarchy)

        # # Add history points if requested
        for i, debug_point in enumerate(self.path.history[::-1]):
            history_hierarchy = self._build_hierarchy_string(
                debug_point.object, f"step_{i}_{debug_point.comment.replace(' ', '_')}"
            )
            if history_hierarchy:
                trace.append(f"{prev} {debug_point.delay - prev_delay}")
                prev = history_hierarchy
                prev_delay = debug_point.delay
                stack_elements.append(history_hierarchy)

        if prev_delay != self.path.delay:
            trace.append(f"{fan_in_hierarchy} {self.path.delay - prev_delay}")

        # # Add fan-out hierarchy (output side of the path)
        # fan_out_hierarchy = self._build_hierarchy_string(self.fan_out, "fan_out")
        # if fan_out_hierarchy:
        #     stack_elements.append(fan_out_hierarchy)

        # # Join with semicolons and add delay
        stack_trace = "\n".join(trace)
        return stack_trace

    def _build_hierarchy_string(self, obj: Object, prefix: str) -> str:
        """
        Build a hierarchical string representation of an Object.

        Args:
            obj: Object to represent
            prefix: Prefix for this part of the hierarchy

        Returns:
            Hierarchical string representation
        """
        parts = [] # [prefix]

        # Add instance path
        for elem in obj.instance_path:
            parts.append(f"{elem.module_name}::{elem.instance_name}")

        # Add signal name and bit position
        signal_part = f"{obj.name}"
        if obj.bit_pos > 0:
            signal_part += f"[{obj.bit_pos}]"
        parts.append(signal_part)

        return ";".join(parts)

@dataclass
class TopPaths:
    """Container for the top critical paths from longest path analysis."""
    paths: List[DataflowPath]


    @classmethod
    def from_mlir(cls, module: circt.ir.Module) -> 'TopPaths':
        """Load top paths from an MLIR string."""
        import mlir.ir
        import mlir.dialects.aig

        with mlir.ir.Context() as ctx:
            analysis = mlir.dialects.aig.LongestPathAnalysis(module)
            paths = analysis.get_top_paths(k=10000)
            return cls(paths=paths)

    @classmethod
    def from_json_file(cls, file_path: Union[str, Path]) -> 'TopPaths':
        """Load top paths from a JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_json_string(cls, json_str: str) -> 'TopPaths':
        """Load top paths from a JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TopPaths':
        """Create TopPaths from a dictionary containing top_paths."""
        data = data[0]
        if "top_paths" in data:
            paths_data = data["top_paths"]
        else:
            # Assume the data is directly the array of paths
            paths_data = data

        paths = [DataflowPath.from_dict(path_data) for path_data in paths_data]
        return cls(paths=paths)

    def get_max_delay(self) -> int:
        """Get the maximum delay among all paths."""
        if not self.paths:
            return 0
        return max(path.path.delay for path in self.paths)

    def get_paths_by_root(self, root_name: str) -> List[DataflowPath]:
        """Get all paths for a specific root module."""
        return [path for path in self.paths if path.root == root_name]

    def get_paths_above_delay(self, min_delay: int) -> List[DataflowPath]:
        """Get all paths with delay above the specified threshold."""
        return [path for path in self.paths if path.path.delay >= min_delay]

    def print_summary(self) -> None:
        """Print a summary of the top paths."""
        print(f"Total paths: {len(self.paths)}")
        if self.paths:
            print(f"Max delay: {self.get_max_delay()}")
            print(f"Min delay: {min(path.path.delay for path in self.paths)}")

            # Group by root
            roots = set(path.root for path in self.paths)
            print(f"Root modules: {', '.join(sorted(roots))}")

            for root in sorted(roots):
                root_paths = self.get_paths_by_root(root)
                print(f"  {root}: {len(root_paths)} paths")

    def to_flamegraph_format(self, include_history: bool = True) -> str:
        """
        Convert all paths to FlameGraph format using detailed format by default.

        Args:
            include_history: If True, include debug history points in the stack trace

        Returns:
            String in FlameGraph format (one line per stack trace with delay)
        """
        lines = []
        for path in self.paths:
            flamegraph_line = path.to_flamegraph_format_detailed(include_history=include_history)
            lines.append(flamegraph_line)
        return '\n'.join(lines)

    def save_flamegraph(self, filename: str, include_history: bool = True) -> None:
        """
        Save paths in FlameGraph format to a file.

        Args:
            filename: Output filename
            include_history: If True, include debug history points in the stack trace
        """
        flamegraph_data = self.to_flamegraph_format(include_history=include_history)
        with open(filename, 'w') as f:
            f.write(flamegraph_data)

def main():
    """Main function to demonstrate usage."""
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python parse_top_paths.py <json_file> [--flamegraph]")
        print("       python parse_top_paths.py <json_file> --flamegraph > output.txt")
        print("")
        print("Options:")
        print("  --flamegraph    Output in FlameGraph format (detailed format with full path info)")
        print("")
        print("Example JSON structure expected:")
        example = {
            "top_paths": [
                {
                    "fan_out": {
                        "instance_path": [
                            {"instance_name": "inst1", "module_name": "ModuleA"}
                        ],
                        "name": "signal_name",
                        "bit_pos": 0
                    },
                    "path": {
                        "fan_in": {
                            "instance_path": [],
                            "name": "input_signal",
                            "bit_pos": 0
                        },
                        "delay": 150,
                        "history": [
                            {
                                "object": {
                                    "instance_path": [],
                                    "name": "intermediate",
                                    "bit_pos": 0
                                },
                                "delay": 75,
                                "comment": "gate delay"
                            }
                        ]
                    },
                    "root": "TopModule"
                }
            ]
        }
        print(json.dumps(example, indent=2))
        sys.exit(1)

    json_file = sys.argv[1]
    flamegraph_mode = len(sys.argv) == 3 and sys.argv[2] == "--flamegraph"

    try:
        top_paths = TopPaths.from_json_file(json_file)

        if flamegraph_mode:
            # Output FlameGraph format (detailed format with full path info)
            print(top_paths.to_flamegraph_format())
        else:
            # Normal summary output
            print(f"Successfully parsed {json_file}")
            top_paths.print_summary()

            # Example of accessing the data
            if top_paths.paths:
                first_path = top_paths.paths[0]
                print(f"\nFirst path details:")
                print(f"  Root: {first_path.root}")
                print(f"  Delay: {first_path.path.delay}")
                print(f"  Fan-out: {first_path.fan_out.name} (bit {first_path.fan_out.bit_pos})")
                print(f"  Fan-in: {first_path.path.fan_in.name} (bit {first_path.path.fan_in.bit_pos})")
                print(f"  History points: {len(first_path.path.history)}")

                # Show FlameGraph format example
                print(f"\nFlameGraph format for first path:")
                print(f"  {first_path.to_flamegraph_format_detailed()}")

    except FileNotFoundError:
        print(f"Error: File '{json_file}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{json_file}': {e}")
        sys.exit(1)
    except KeyError as e:
        print(f"Error: Missing required field in JSON: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
