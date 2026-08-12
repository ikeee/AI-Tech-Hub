"""老照片修复 常驻 Worker：模型只加载一次，循环处理 stdin JSON 任务。

协议（每行一个 JSON）：
  -> {"input": "...", "text": "...", "params": {}, "out": "..."}
  <- {"type": "ready" | "loaded" | "done", "out": "..."} 或 {"type": "error", "error": "..."}

注意：Windows 下 stdin/stdout 必须 reconfigure 为 utf-8（Node 发送 UTF-8）。
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
        # 模型只加载一次（RealESRGAN + CodeFormer + 人脸检测，首次约 1-3 分钟）
        from main import get_restorer
        get_restorer()
        print(json.dumps({"type": "loaded"}), flush=True)
    except Exception:
        print(json.dumps({"type": "error", "error": "模型加载失败，请查看服务端日志"}), flush=True)
        traceback.print_exc()
        return

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            task = json.loads(line)
            from main import run
            out = run(task.get("input", ""), task.get("params", {}), task.get("out", "output"))
            print(json.dumps({"type": "done", "out": out}), flush=True)
        except Exception as e:
            print(json.dumps({"type": "error", "error": str(e)}), flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    main()