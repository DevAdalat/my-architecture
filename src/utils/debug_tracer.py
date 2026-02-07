import datetime
import json

import torch


class DebugTracer:
    def __init__(self):
        self.logs = {}
        self.active = True

    def update(self, step_name, variables=None):
        if not self.active:
            return

        timestamp = datetime.datetime.now().isoformat()

        # Capture memory stats
        memory_stats = {}
        if torch.cuda.is_available():
            memory_stats = {
                "allocated": f"{torch.cuda.memory_allocated() / 1024**2:.2f} MB",
                "reserved": f"{torch.cuda.memory_reserved() / 1024**2:.2f} MB",
                "max_allocated": f"{torch.cuda.max_memory_allocated() / 1024**2:.2f} MB",
            }

        # Capture variable metadata (avoid storing full tensors)
        var_stats = {}
        if variables:
            for k, v in variables.items():
                if isinstance(v, torch.Tensor):
                    var_stats[k] = {
                        "type": "Tensor",
                        "shape": str(list(v.shape)),
                        "dtype": str(v.dtype),
                        "device": str(v.device),
                        "requires_grad": v.requires_grad,
                    }
                elif isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                    var_stats[k] = {
                        "type": f"List[Tensor] (len={len(v)})",
                        "shape_0": str(list(v[0].shape)),
                    }
                else:
                    var_stats[k] = str(v)[:200]  # Truncate long strings

        self.logs[step_name] = {
            "timestamp": timestamp,
            "memory": memory_stats,
            "variables": var_stats,
        }

    def clear(self):
        self.logs = {}

    def dump(self, filename="crash_dump.json"):
        print(f"\n[DebugTracer] Dumping crash logs to {filename}...")
        try:
            with open(filename, "w") as f:
                json.dump(self.logs, f, indent=2)
            print("[DebugTracer] Dump successful.")
        except Exception as e:
            print(f"[DebugTracer] Failed to dump logs: {e}")
            # Fallback print
            print(json.dumps(self.logs, indent=2))


# Global instance
tracer = DebugTracer()
