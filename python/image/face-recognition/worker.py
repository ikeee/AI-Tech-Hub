"""人脸识别/验证常驻 Worker（insightface 模型只加载一次）。

协议（每行一个 JSON，与 sd-turbo worker 一致）：
  -> {"mode": "recognition", "input": "path"} 或 {"mode": "verification", "input": "p1", "second": "p2"}
  <- {"type": "ready" | "loaded" | "done", "result": {...}} 或 {"type": "error", "error": "..."}
"""

import json
import sys
import traceback


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
        sys.stdin.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print(json.dumps({"type": "ready"}), flush=True)
    try:
        from main import get_app
        get_app()
        print(json.dumps({"type": "loaded"}), flush=True)
    except Exception:
        print(json.dumps({"type": "error", "error": "模型加载失败（需 pip install insightface onnxruntime，首次会下载 buffalo_l 约 300MB）"}), flush=True)
        traceback.print_exc()
        return

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            task = json.loads(line)
            from main import recognize, verify
            if task.get("mode") == "verification":
                result = verify(task["input"], task["second"])
            else:
                result = recognize(task["input"])
            print(json.dumps({"type": "done", "result": result}), flush=True)
        except Exception as e:
            print(json.dumps({"type": "error", "error": str(e)}), flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    main()
