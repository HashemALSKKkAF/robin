import os
import re
import logging
import threading
from bs4 import BeautifulSoup
from pathlib import Path
from typing import List, Optional


class HtmlToTxt:
    """
    Convert rendered.html → snap.txt inside:
    ~/Obscura/data/loader_pages/<hash>_<timestamp>/
    """

    CONST_PATH = str(Path.home() / "Obscura" / "data" / "loader_pages")
    LOG_PATH = Path.home() / "Obscura" / "logs" / "html_txt.log"

    _lock = threading.Lock()

    # ----------------------
    # Initialize logging
    # ----------------------
    logging_initialized = False

    @staticmethod
    def _init_logging():
        """Initialize the logger once."""
        if HtmlToTxt.logging_initialized:
            return

        log_dir = HtmlToTxt.LOG_PATH.parent
        log_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            filename=str(HtmlToTxt.LOG_PATH),
            filemode="a",
            format="%(asctime)s | %(threadName)s | %(levelname)s | %(message)s",
            level=logging.INFO,
        )

        HtmlToTxt.logging_initialized = True

    # ----------------------

    @staticmethod
    def _log(level: str, msg: str):
        HtmlToTxt._init_logging()
        with HtmlToTxt._lock:
            if level == "info":
                logging.info(msg)
            elif level == "warn":
                logging.warning(msg)
            elif level == "error":
                logging.error(msg)

    # ----------------------

    @staticmethod
    def _extract_text_from_html(html_content: str) -> str:
        """Extract visible text from HTML."""
        soup = BeautifulSoup(html_content, "html.parser")

        for tag in soup(["script", "style", "noscript"]):
            tag.extract()

        text = soup.get_text(separator="\n")
        text = re.sub(r"\n\s*\n", "\n\n", text)
        return text.strip()

    @staticmethod
    def _find_folders_for_hash(hash_value: str) -> List[Path]:
        """Locate all folders matching <hash>_<timestamp>."""
        base = Path(HtmlToTxt.CONST_PATH)
        if not base.exists():
            return []

        folders = [
            entry for entry in base.iterdir()
            if entry.is_dir() and entry.name.startswith(hash_value)
        ]

        folders.sort(key=lambda p: p.name.split("_")[1], reverse=True)
        return folders

    @staticmethod
    def _process_folder(folder: Path) -> Optional[str]:
        """Convert rendered.html → snap.txt unless it already exists."""

        snap_path = folder / "snap.txt"
        html_path = folder / "rendered.html"

        # Skip if snap exists
        if snap_path.exists():
            msg = f"[SKIP] snap exists | folder={folder}"
            HtmlToTxt._log("info", msg)
            return msg

        if not html_path.exists():
            msg = f"[NO_HTML] missing rendered.html | folder={folder}"
            HtmlToTxt._log("warn", msg)
            return msg

        # Read HTML
        try:
            with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
                html = f.read()
        except Exception as e:
            msg = f"[ERR_READ] {e} | file={html_path}"
            HtmlToTxt._log("error", msg)
            return msg

        # Extract text
        text = HtmlToTxt._extract_text_from_html(html)

        # Write snap.txt
        try:
            with HtmlToTxt._lock:
                with open(snap_path, "w", encoding="utf-8") as f:
                    f.write(text)

            msg = f"[CREATED] snap.txt created | folder={folder}"
            HtmlToTxt._log("info", msg)
            return msg

        except Exception as e:
            msg = f"[ERR_WRITE] {e} | file={snap_path}"
            HtmlToTxt._log("error", msg)
            return msg

    @staticmethod
    def run(hash_value: str) -> List[str]:
        """
        Main:
        - find folders with same hash
        - sort newest → oldest
        - create snap.txt where missing
        - log each step
        """
        HtmlToTxt._init_logging()

        msg = f"[RUN] Processing hash={hash_value}"
        HtmlToTxt._log("info", msg)

        results = []
        folders = HtmlToTxt._find_folders_for_hash(hash_value)

        if not folders:
            msg = f"[NO_FOLDERS] hash={hash_value}"
            HtmlToTxt._log("warn", msg)
            return [msg]

        for folder in folders:
            result = HtmlToTxt._process_folder(folder)
            results.append(result)

        return results
