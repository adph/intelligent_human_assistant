import asyncio
import os
import signal
import subprocess
import time
import uuid
from typing import Optional

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession

# Allow port override without editing code (handy when composing multiple MCP servers).
MCP_PORT = int(os.getenv("TERMINAL_MCP_PORT", "3002"))
mcp = FastMCP(name="terminal", port=MCP_PORT)

user_home = os.getenv("HOME") or os.getcwd()
proc: Optional[subprocess.Popen[str]] = None


def _normalize_cwd(cwd: str) -> str:
    if not cwd:
        return user_home
    if "~" in cwd:
        cwd = os.path.expanduser(cwd)
    return cwd


def _start_bash(cwd: str) -> subprocess.Popen[str]:
    '''
    Start an interactive bash process.

    We merge stderr into stdout so callers see errors in the streamed/returned output.
    '''
    return subprocess.Popen(
        ["/bin/bash"],
        cwd=cwd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # line-buffered
    )


@mcp.tool()
async def initiate_terminal(cwd: str = "") -> str:
    '''
    Initiate (or reset) an interactive bash terminal session.

    Args:
        cwd: Directory where the terminal is initiated. Defaults to user's HOME.
    '''
    global proc
    cwd = _normalize_cwd(cwd)

    if not os.path.isdir(cwd):
        return f"{cwd} is not a directory"

    # If already running, reset it (simple + predictable for demos).
    if proc is not None:
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    proc = _start_bash(cwd)
    return f"Terminal initiated (cwd={cwd})"


@mcp.tool()
async def terminate_terminal() -> str:
    '''Terminate the current terminal session (if any).'''
    global proc
    if proc is None:
        return "No terminal is open."
    try:
        proc.terminate()
        proc.wait(timeout=2)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    proc = None
    return "Terminal terminated."


async def _ensure_terminal(ctx: Optional[Context[ServerSession, None]] = None) -> None:
    '''Ensure we have a running bash process. If not, start one in HOME.'''
    global proc
    if proc is None or proc.poll() is not None:
        if ctx is not None:
            await ctx.info("No active terminal session found; starting one in HOME.")
        await initiate_terminal("")


@mcp.tool()
async def execute_command(
    cmd: str,
    ctx: Context[ServerSession, None],
    timeout_sec: int = 30,
) -> str:
    '''
    Execute a bash command in the current (persistent) terminal session.

    - Streams output line-by-line using MCP logging notifications (ctx.info),
      so clients that support logs/progress can render output as it arrives.
    - Returns the full captured output as a single string as well.

    Args:
        cmd: Bash command to run.
        timeout_sec: Safety timeout to prevent hanging forever.
    '''
    global proc
    await _ensure_terminal(ctx)

    assert proc is not None and proc.stdin is not None and proc.stdout is not None

    # Unique marker to detect end-of-command output.
    marker = f"__MCP_CMD_DONE_{uuid.uuid4().hex}__"

    # Write the command + marker.
    to_send = cmd.rstrip("\n") + "\n" + f"echo {marker}\n"
    proc.stdin.write(to_send)
    proc.stdin.flush()

    await ctx.info(f"$ {cmd}")

    lines: list[str] = []
    deadline = time.time() + max(1, int(timeout_sec))

    while True:
        # Timeout check (prevents deadlocks if something never prints marker).
        if time.time() > deadline:
            await ctx.info(f"[timeout after {timeout_sec}s] Attempting to interrupt command…")
            try:
                proc.send_signal(signal.SIGINT)
            except Exception:
                pass
            break

        # Read stdout without blocking the event loop.
        line = await asyncio.to_thread(proc.stdout.readline)
        if not line:
            break

        if marker in line:
            break

        line = line.rstrip("\n")
        lines.append(line)

        # "Streaming response": emit incremental output as log notifications.
        if line.strip():
            await ctx.info(line)

    out = "\n".join(lines).strip()
    return out if out else "(no output)"


# Backwards-compatible alias for older clients (if you used run_command in P1).
@mcp.tool()
async def run_command(
    command: str,
    ctx: Context[ServerSession, None],
    timeout_sec: int = 30,
) -> str:
    '''Alias for execute_command(command, ...).'''
    return await execute_command(command, ctx, timeout_sec)


if __name__ == "__main__":
    # Streamable HTTP => Open WebUI can connect at http://localhost:<port>/mcp
    mcp.run(transport="streamable-http")
