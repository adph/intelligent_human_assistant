"""
Hybrid MCP Orchestration Layer for Voice-Driven UI Interaction

This module provides a combined architecture built from:
- mcp-agent: manages deterministic workflow sequencing
- FastMCP: lightweight server with streaming HTTP support (default port 3001)

Primary capabilities include:
1. Capturing current screen content
2. Running OCR to identify visible UI text and regions
3. Associating numeric labels with detected elements
4. Executing automated pointer actions when instructed verbally
"""

import os

# =============================================================================
# RUNTIME ENVIRONMENT PREPARATION
# =============================================================================
# Limit concurrency for predictable CPU usage
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
import torch

# Ensure CPU-based execution for OCR + workflow logic
torch.set_default_device("cpu")
torch.set_num_threads(1)

# =============================================================================
# CORE IMPORTS
# =============================================================================
from mcp.server.fastmcp import FastMCP
from mcp_agent.app import MCPApp
from mcp_agent.executor.workflow import Workflow, WorkflowResult
from datetime import timedelta

import tempfile
import json
import re
import glob
import logging
from PIL import ImageGrab
from paddleocr import PaddleOCR
from gtts import gTTS
from playsound3 import playsound
import threading
from typing import List, Dict, Any, Optional
from pathlib import Path
import pyautogui

# =============================================================================
# TEXT-TO-SPEECH HELPERS
# =============================================================================

def speak_text_out_loud(message: str) -> None:
    """Generate speech audio asynchronously using Google TTS."""

    def _run_tts():
        try:
            print(f"[TTS] Speaking: {message[:50]}...")
            temp_dir = tempfile.gettempdir()
            audio_path = os.path.join(temp_dir, f"tts_{hash(message)}.mp3")

            tts_obj = gTTS(text=message, lang="en")
            tts_obj.save(audio_path)
            playsound(audio_path)

            # Clean up audio file after playback
            try:
                os.remove(audio_path)
            except:
                pass

        except Exception as exc:
            print(f"[TTS Error] {exc}")

    threading.Thread(target=_run_tts, daemon=True).start()


# =============================================================================
# SCREEN & RESOLUTION DISCOVERY
# =============================================================================

screen_w, screen_h = pyautogui.size()
print(f"[Display] Screen resolution: {screen_w} × {screen_h}")

_reference_capture = pyautogui.screenshot()
img_w, img_h = _reference_capture.size
print(f"[Display] Raw screenshot resolution: {img_w} × {img_h}")

scale_x_factor = screen_w / img_w
scale_y_factor = screen_h / img_h
print(f"[Display] Scaling factors → X: {scale_x_factor}, Y: {scale_y_factor}")

# =============================================================================
# LOGGER CONFIGURATION
# =============================================================================

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

mcp_port = int(os.getenv("P5_MCP_PORT", "3001"))

# =============================================================================
# SERVER BOOTSTRAP
# =============================================================================

mcp_server = FastMCP("P5", port=mcp_port)
mcp_engine = MCPApp(name="workflow-engine")

# =============================================================================
# ENVIRONMENT FILE UTILITIES
# =============================================================================

def load_env_vars() -> dict:
    """
    Load key/value pairs from a .env file (if present) into a dictionary.

    Only lines matching KEY=VALUE are considered; comments are ignored.
    """
    env_file = Path(".env")
    collected = {}

    if env_file.exists():
        for raw_line in env_file.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            collected[key.strip()] = value.strip().strip('"').strip("'")

    return collected


# =============================================================================
# COORDINATE MAPPING (P5 REQUIREMENT)
# =============================================================================

def _coordinates_json_path() -> str:
    """Resolve the path to the coordinate mapping file.

    Priority:
    1) COORDINATES_JSON_PATH in .env
    2) ./coordinates.json next to this script
    """
    settings = load_env_vars()
    configured = settings.get("COORDINATES_JSON_PATH")
    if configured:
        return configured
    return str(Path(__file__).with_name("coordinates.json"))


def _load_coordinates(path: str) -> Dict[str, Dict[str, int]]:
    """Load mapping file; if missing, create an empty one."""
    try:
        p = Path(path)
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("{}")
            return {}
        raw = p.read_text().strip() or "{}"
        data = json.loads(raw)
        if not isinstance(data, dict):
            return {}
        # normalize values
        cleaned: Dict[str, Dict[str, int]] = {}
        for k, v in data.items():
            if isinstance(v, dict) and "x" in v and "y" in v:
                try:
                    cleaned[str(k)] = {"x": int(v["x"]), "y": int(v["y"])}
                except Exception:
                    continue
        return cleaned
    except Exception:
        return {}


def _save_coordinates(path: str, mapping: Dict[str, Dict[str, int]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(mapping, indent=2))


# =============================================================================
# MCP TOOLS: Coordinate-based UI automation (P5)
# =============================================================================

@mcp_server.tool()
async def ui_click(element_name: str) -> str:
    """Click a named UI element using coordinate mapping.

    Reads coordinates from COORDINATES_JSON_PATH (in .env) or ./coordinates.json.
    Returns a JSON string with success/failure.
    """
    try:
        path = _coordinates_json_path()
        mapping = _load_coordinates(path)
        if element_name not in mapping:
            return json.dumps({
                "status": "error",
                "error": f"Unknown element_name '{element_name}'. Add it to {path}.",
                "known_keys": sorted(mapping.keys())
            })

        coords = mapping[element_name]
        x, y = int(coords["x"]), int(coords["y"])
        pyautogui.click(x, y)
        return json.dumps({"status": "ok", "element_name": element_name, "x": x, "y": y})
    except Exception as exc:
        return json.dumps({"status": "error", "error": str(exc)})


@mcp_server.tool()
async def ui_type(element_name: str, text: str, press_enter: bool = False) -> str:
    """Focus a named UI element and type text.

    Behavior:
    1) ui_click(element_name) to focus
    2) type the provided text
    3) optionally press Enter
    """
    try:
        click_res = json.loads(await ui_click(element_name))
        if click_res.get("status") != "ok":
            return json.dumps(click_res)

        # Type text
        pyautogui.write(text, interval=0.01)
        if press_enter:
            pyautogui.press("enter")

        return json.dumps({"status": "ok", "element_name": element_name, "typed": True, "press_enter": press_enter})
    except Exception as exc:
        return json.dumps({"status": "error", "error": str(exc)})


@mcp_server.tool()
async def ui_record_coordinate(element_name: str) -> str:
    """Record the current mouse position into the coordinate map.

    This is optional (not required by P5), but makes setup fast:
    - hover mouse over the target UI element
    - call ui_record_coordinate("browser_close_last_tab")
    """
    try:
        path = _coordinates_json_path()
        mapping = _load_coordinates(path)
        x, y = pyautogui.position()
        mapping[str(element_name)] = {"x": int(x), "y": int(y)}
        _save_coordinates(path, mapping)
        return json.dumps({"status": "ok", "saved_to": path, "element_name": element_name, "x": int(x), "y": int(y)})
    except Exception as exc:
        return json.dumps({"status": "error", "error": str(exc)})

# =============================================================================
# OCR SINGLETON WRAPPER
# =============================================================================

_ocr_singleton = None

def acquire_ocr_engine() -> PaddleOCR:
    """Return a lazily-initialized OCR engine instance."""
    global _ocr_singleton

    if _ocr_singleton is None:
        _ocr_singleton = PaddleOCR(
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="PP-OCRv5_mobile_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    return _ocr_singleton


def pull_ascii_digits(raw_text: str) -> str:
    """Extract only 0–9 ASCII digits from the supplied string."""
    return "".join(ch for ch in raw_text if ch.isascii() and ch.isdigit())

# =============================================================================
# WORKFLOW TEMPLATE: BasicScreenshotWorkflow
# (Renamed from VanillaWorkflow; comments restyled)
# =============================================================================

@mcp_engine.workflow
class BasicScreenshotWorkflow(Workflow[dict]):
    """
    A minimal demonstration workflow showing:
    - reading configuration
    - capturing a screenshot
    - announcing workflow status

    This reference implementation is intentionally simple and
    should be used as a guide for constructing more complex workflows.
    """

    screenshot_root = ""

    @mcp_engine.workflow_task(schedule_to_close_timeout=timedelta(minutes=1))
    async def fetch_screenshot_directory(self) -> dict:
        """Retrieve screenshot storage path from environment config."""
        settings = load_env_vars()
        self.screenshot_root = settings.get("PATH_TO_SCREENSHOT")

        log.info(f"[BasicWorkflow] Screenshot directory set to: {self.screenshot_root}")
        return settings

    @mcp_engine.workflow_task(schedule_to_close_timeout=timedelta(minutes=2))
    async def capture_image(self) -> str:
        """Create and save a screenshot image to the configured directory."""
        saved_path = ""

        if self.screenshot_root:
            saved_path = os.path.join(
                self.screenshot_root,
                f"screencap_{id(self)}.png"
            )
            img = ImageGrab.grab()
            img.save(saved_path, "PNG")
            log.info(f"[BasicWorkflow] Screenshot saved → {saved_path}")
        else:
            log.error("[BasicWorkflow] Missing .env configuration for screenshot directory.")

        return saved_path

    @mcp_engine.workflow_task(schedule_to_close_timeout=timedelta(minutes=2))
    async def speak_status(self, message: str) -> dict:
        """Announce a status message using speech synthesis."""
        speak_text_out_loud(message)
        return {"spoken": message, "state": "ok"}

    @mcp_engine.workflow_run
    async def run(self, target: str) -> WorkflowResult[dict]:
        """
        Execute the full workflow:
        1. Identify where screenshots should be saved.
        2. Capture screen image.
        3. Speak success/failure outcome.
        """
        step1 = await self.fetch_screenshot_directory()
        step2 = await self.capture_image()

        time.sleep(0.2)

        outcome = "completed" if step2 else "failure"
        step3 = await self.speak_status(outcome)

        return WorkflowResult(
            value={
                "workflow": "BasicScreenshotWorkflow",
                "target": target,
                "steps": [step1, step2, step3],
                "status": outcome,
                "detail": f"Workflow finished {'successfully' if step2 else 'unsuccessfully'}."
            }
        )


# =============================================================================
# TOOL: Execute BasicScreenshotWorkflow
# =============================================================================

@mcp_server.tool()
async def basic_workflow_tool(target=None) -> str:
    """Public entry point invoking BasicScreenshotWorkflow."""
    try:
        wf = BasicScreenshotWorkflow()
        result = await wf.run(target)
        return json.dumps(result.value, indent=2)

    except Exception as exc:
        log.error(f"[BasicWorkflow Tool Error] {exc}")
        return json.dumps({"status": "error", "error": str(exc)})


# =============================================================================
# WORKFLOW: ClickWorkflow (Cosmetically refactored)
# =============================================================================

@mcp_engine.workflow
class ClickWorkflow(Workflow[dict]):
    """
    Workflow that:
    1. Loads OCR metadata from disk
    2. Locates a UI element that matches a search phrase or number
    3. Clicks on the selected item
    4. Announces selected action
    """

    @mcp_engine.workflow_task(schedule_to_close_timeout=timedelta(minutes=1))
    async def load_metadata(self, json_path: str) -> dict:
        """Read OCR JSON output generated from a previous screen capture."""
        try:
            with open(json_path, "r") as f:
                parsed = json.load(f)

            log.info(f"[ClickWorkflow] Loaded OCR metadata from {json_path}")
            return parsed

        except Exception as exc:
            log.error(f"[ClickWorkflow] Failed reading JSON {json_path}: {exc}")
            raise

    @mcp_engine.workflow_task(schedule_to_close_timeout=timedelta(minutes=2))
    async def identify_target(self, ocr_blob: dict, query: str) -> dict:
        """Search through OCR mappings to find the element matching the given query."""

        mappings = ocr_blob.get("mappings", [])
        lowered = query.lower()
        chosen = None

        for item in mappings:
            text_val = item.get("text", "").lower()
            num_val = item.get("number", "").lower()

            if lowered in text_val or lowered == num_val:
                chosen = item
                break

        if not chosen:
            raise ValueError(f"No element found matching '{query}'.")

        log.info(f"[ClickWorkflow] Target resolved → {chosen}")
        return chosen

    @mcp_engine.workflow_task(schedule_to_close_timeout=timedelta(minutes=2))
    async def execute_click(self, target_info: dict) -> dict:
        """Perform the mouse click operation on the selected UI element."""
        cx = target_info.get("center_x")
        cy = target_info.get("center_y")

        if cx is None or cy is None:
            raise ValueError("Missing coordinate data for click operation.")

        pyautogui.click(int(cx * scale_x_factor), int(cy * scale_y_factor))
        log.info(f"[ClickWorkflow] Clicked at scaled coords: ({cx},{cy})")

        return {"clicked": True, "raw_center": (cx, cy)}

    @mcp_engine.workflow_task(schedule_to_close_timeout=timedelta(minutes=2))
    async def verbalize(self, phrase: str) -> dict:
        """Speak a confirmation or error message aloud."""
        speak_text_out_loud(phrase)
        return {"spoken": phrase, "ok": True}

    @mcp_engine.workflow_run
    async def run(self, query: str, metadata_path: str) -> WorkflowResult[dict]:
        """Coordinated execution: metadata → selection → click → announcement."""
        step1 = await self.load_metadata(metadata_path)
        step2 = await self.identify_target(step1, query)
        step3 = await self.execute_click(step2)
        step4 = await self.verbalize(f"Executed click for {query}")

        return WorkflowResult(
            value={
                "workflow": "ClickWorkflow",
                "query": query,
                "steps": [step1, step2, step3, step4],
                "status": "done",
                "message": f"Click action executed for '{query}'."
            }
        )

# =============================================================================
# TOOL: ClickWorkflow Launcher
# =============================================================================

@mcp_server.tool()
async def click_workflow_tool(target: str) -> str:
    """
    Tool endpoint to trigger the click workflow.
    Automatically finds the latest OCR metadata and forwards both
    the target string and metadata path to ClickWorkflow.
    """
    try:
        settings = load_env_vars()
        data_folder = settings.get("DATA_DIR", ".")

        # Locate OCR metadata files
        json_candidates = glob.glob(os.path.join(data_folder, "screenshot_*_ocr.json"))
        if not json_candidates:
            return json.dumps({
                "status": "error",
                "error": "No OCR metadata files found — run capture_screen_with_numbers_tool() first."
            })

        # Most recently created file
        latest_metadata = max(json_candidates, key=os.path.getctime)
        log.info(f"[ClickWorkflow Tool] Using metadata file: {latest_metadata}")

        workflow = ClickWorkflow()
        output = await workflow.run(target, latest_metadata)
        return json.dumps(output.value, indent=2)

    except Exception as exc:
        log.error(f"[ClickWorkflow Tool Error] {exc}")
        return json.dumps({"status": "error", "error": str(exc)})


# =============================================================================
# WORKFLOW: CaptureScreenWithNumbers
# =============================================================================

@mcp_engine.workflow
class CaptureScreenWithNumbers(Workflow[dict]):
    """
    Workflow that:
    1. Announces steps (e.g., "show numbers")
    2. Captures a fresh screenshot
    3. Applies OCR to extract UI text + polygon regions
    4. Maps numeric sequences to coordinates
    5. Saves mapping JSON + screenshot
    """

    screenshot_root = ""
    metadata_output_path = ""

    @mcp_engine.workflow_task(schedule_to_close_timeout=timedelta(minutes=2))
    async def announce_step(self, phrase: str) -> dict:
        """Speak a workflow stage command aloud."""
        speak_text_out_loud(phrase)
        time.sleep(5)
        return {"phrase": phrase, "ack": True}

    @mcp_engine.workflow_task(schedule_to_close_timeout=timedelta(minutes=2))
    async def process_screen(self) -> dict:
        """Perform screenshot capture, OCR analysis, and mapping extraction."""

        settings = load_env_vars()
        self.screenshot_root = settings.get("PATH_TO_SCREENSHOT", ".")
        data_folder = settings.get("DATA_DIR", ".")

        # Unique ID for naming screenshot + OCR metadata
        snap_id = int(time.time())
        screenshot_path = os.path.join(self.screenshot_root, f"screenshot_{snap_id}.png")

        # Capture on-screen content
        cap = pyautogui.screenshot()
        cap.save(screenshot_path, "PNG")
        log.info(f"[Capture] Saved screenshot → {screenshot_path}")

        self.capture_id = snap_id
        ocr_engine = acquire_ocr_engine()
        ocr_results = ocr_engine.ocr(screenshot_path)

        extracted_elements = []

        # Parse OCR output
        if ocr_results and ocr_results[0]:
            data_block = ocr_results[0]

            # Handle detection structures whether dict-like or attribute-based
            if hasattr(data_block, "get") or isinstance(data_block, dict):
                text_entries = data_block.get("rec_texts", []) if hasattr(data_block, "get") else getattr(data_block, "rec_texts", [])
                polygon_entries = data_block.get("rec_polys", []) if hasattr(data_block, "get") else getattr(data_block, "rec_polys", [])
                confidence_scores = data_block.get("rec_scores", []) if hasattr(data_block, "get") else getattr(data_block, "rec_scores", [])

                # Iterate through results
                for idx, (txt, poly) in enumerate(zip(text_entries, polygon_entries)):
                    try:
                        # Some OCR models may return fewer scores than entries
                        score_val = confidence_scores[idx] if idx < len(confidence_scores) else 0.0

                        if hasattr(poly, "__iter__") and len(poly) >= 4:
                            pts = poly.tolist() if hasattr(poly, "tolist") else poly

                            # Compute centroid
                            centroid_x = sum(p[0] for p in pts) / len(pts)
                            centroid_y = sum(p[1] for p in pts) / len(pts)

                            # Extract digits (if present)
                            numeric_part = pull_ascii_digits(txt)

                            # Confidence filtering
                            if score_val > 0.5:
                                extracted_elements.append({
                                    "number": numeric_part if numeric_part else "",
                                    "text": txt,
                                    "center_x": centroid_x,
                                    "center_y": centroid_y,
                                    "confidence": score_val
                                })
                    except:
                        continue

        # Save OCR mapping JSON
        json_name = f"screenshot_{self.capture_id}_ocr.json"
        self.metadata_output_path = os.path.join(data_folder, json_name)

        with open(self.metadata_output_path, "w") as f:
            json.dump({"mappings": extracted_elements}, f, indent=2)

        log.info(f"[Capture] Mapped elements count: {len(extracted_elements)}")
        log.info(f"[Capture] Metadata saved at → {self.metadata_output_path}")

        return {
            "screenshot": screenshot_path,
            "metadata": self.metadata_output_path,
            "detected_count": len(extracted_elements),
        }

    @mcp_engine.workflow_run
    async def run(self) -> WorkflowResult[dict]:
        """Execute the multi-step OCR pipeline with three verbal stages."""

        step1 = await self.announce_step("start listening")
        time.sleep(4)

        step2 = await self.announce_step("show numbers")

        step3 = await self.process_screen()

        step4 = await self.announce_step("stop listening")

        return WorkflowResult(
            value={
                "workflow": "CaptureScreenWithNumbers",
                "status": "Success",
                "summary": f"Screen captured and {step3['detected_count']} numbered elements extracted.",
                "details": {
                    "screenshot_path": step3["screenshot"],
                    "metadata_file": step3["metadata"],
                    "elements_found": step3["detected_count"],
                    "next_action": "Use click_workflow_tool() to select a UI element."
                }
            }
        )


# =============================================================================
# TOOL: CaptureScreenWithNumbers Launcher
# =============================================================================

@mcp_server.tool()
async def capture_screen_with_numbers_tool():
    """
    Tool endpoint that triggers the numbered-screen capture workflow.
    Runs the full OCR mapping pipeline and returns structured results.
    """
    try:
        wf = CaptureScreenWithNumbers()
        result = await wf.run()
        return json.dumps(result.value, indent=2)

    except Exception as exc:
        log.error(f"[CaptureScreen Tool Error] {exc}")
        return json.dumps({"status": "error", "error": str(exc)})


# =============================================================================
# TOOL: Direct Speech-Echo Utility
# =============================================================================

@mcp_server.tool()
async def echo_tool(command: str) -> str:
    """
    Simple utility that:
    - Speaks the given command aloud
    - Returns a structured echo response

    Useful for testing voice connectivity and verifying browser events.
    """
    speak_text_out_loud(command)
    return f"[Echo] Spoken: '{command}'"


# =============================================================================
# MAIN LAUNCH POINT
# =============================================================================

if __name__ == "__main__":
    log.info("=" * 60)
    log.info("Hybrid Server: mcp-agent orchestration + FastMCP streamable HTTP")
    log.info("=" * 60)
    log.info(f"Access endpoint at → http://127.0.0.1:{mcp_port}/mcp")
    log.info("=" * 60)

    mcp_server.run(transport="streamable-http")
