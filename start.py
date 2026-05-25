from __future__ import annotations

import socket

from backend.api_server import API_HOST, API_PORT, main


def _port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


if __name__ == "__main__":
    app_url = f"http://{API_HOST}:{API_PORT}"
    if _port_open(API_HOST, API_PORT):
        print(f"[img-search] 服务已在运行：{app_url}", flush=True)
        raise SystemExit(0)

    print(f"[img-search] 正在启动服务：{app_url}", flush=True)
    print("[img-search] 这是前台常驻进程，终端会保持占用；按 Ctrl+C 停止服务。", flush=True)
    main()
