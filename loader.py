#!/usr/bin/env python3
"""
tor_crawler_torbrowser.py

Selenium + Tor Browser crawler for .onion pages with improvements:
- Uses Tor Browser binary (better fingerprint & DNS behavior than system Firefox)
- Auto-detects Tor SOCKS port (9050 or 9150)
- Optional Tor Control (Stem) NEWNYM circuit rotation
- User-Agent rotation
- Human-like scrolling & basic mouse interactions
- Extracts links via executed JS (live DOM) + BeautifulSoup fallback
- Captcha / JS-challenge detection scaffolding
- Saves HTML, screenshot, metadata, and extracted links
- Keeps a global JSONL index of seen links (like your original)
"""

import os
import sys
import time
import random
import socket
import json
import logging
import signal
import hashlib
import re
from datetime import datetime
from urllib.parse import urlparse, urljoin

# Selenium + Firefox
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import TimeoutException, WebDriverException, JavascriptException

# BeautifulSoup fallback
from bs4 import BeautifulSoup, builder
from login_manager import LoginManager


# Optional: stem for Tor control (circuit rotation)
try:
    from stem import Signal
    from stem.control import Controller
    STEM_AVAILABLE = True
except Exception:
    STEM_AVAILABLE = False

# ===================== Configuration =====================
class TorRendererConfig:
    # --- Tor / Tor Browser ---
    TOR_SOCKS_HOST = "127.0.0.1"
    # We'll auto-detect between these two common ports
    TOR_SOCKS_PORTS = (9050, 9150)
    TOR_CONTROL_PORT = 9051  # typical for system tor; change if different
    TOR_CONTROL_PASSWORD = None  # set a string if you configured ControlPort with a password in torrc
    TOR_BROWSER_BINARY = "/home/kali/.local/share/torbrowser/tbb/x86_64/tor-browser/Browser/firefox"  # <-- REPLACE: path to Tor Browser's Firefox binary (platform dependent)
    # Example Linux Tor Browser binary usually:
    # ~/tor-browser_en-US/Browser/firefox or /opt/tor-browser/Browser/firefox

    # --- Operation ---
    HEADLESS = False  # headful is often less-detectable; set True if you must run headless
    PAGE_LOAD_TIMEOUT = 60
    POST_LOAD_SLEEP = 3
    RETRIES = 3
    RETRY_DELAY = 6
    RATE_LIMIT_SLEEP = 3
    OUTPUT_BASE = os.path.expanduser("~/Obscura/data/loader_pages")
    GLOBAL_INDEX = os.path.expanduser("~Obscura/data/loader_pages/links_index.jsonl")
    MAX_HTML_BYTES = 8 * 1024 * 1024  # 8 MB

    # --- UA Rotation (small curated list; extend as you like) ---
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:119.0) Gecko/20100101 Firefox/119.0",
        "Mozilla/5.0 (X11; Linux x86_64; rv:140.0) Gecko/20100101 Firefox/140.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.4 Safari/605.1.15"
    ]

    # --- Download and suspicious extensions ---
    DOWNLOAD_EXTENSIONS = (
        '.zip', '.rar', '.7z', '.exe', '.msi', '.deb', '.rpm',
        '.tar', '.gz', '.bz2', '.iso', '.doc', '.docx', '.pdf',
        '.xls', '.xlsx', '.ppt', '.pptx', '.apk', '.bin', '.dll'
    )

    # Simple regex to find href/src in raw HTML for suspicious link scanning
    LINK_PATTERN = re.compile(r'(?:href|src)\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE)

# ===================== Logger =====================
class LoggerManager:
    @staticmethod
    def setup_logger(log_path):
        logger = logging.getLogger("tor_renderer_tb")
        logger.setLevel(logging.DEBUG)
        if not logger.hasHandlers():
            fh = logging.FileHandler(log_path, encoding='utf-8')
            fh.setLevel(logging.DEBUG)
            sh = logging.StreamHandler(sys.stdout)
            sh.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            sh.setFormatter(formatter)
            logger.addHandler(fh)
            logger.addHandler(sh)
        return logger

# ===================== Utilities =====================
def safe_filename(s):
    return re.sub(r'[^A-Za-z0-9_.-]', '_', s)

def sha256_text(t):
    return hashlib.sha256(t.encode('utf-8')).hexdigest()

def atomic_write_text(path, text, encoding='utf-8'):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding=encoding) as f:
        f.write(text)
    os.replace(tmp, path)

def atomic_write_bytes(path, data):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)

# ===================== Tor helpers =====================
def find_open_socks_port(host="127.0.0.1", ports=(9050, 9150), timeout=0.5):
    for p in ports:
        try:
            with socket.create_connection((host, p), timeout=timeout):
                return p
        except Exception:
            continue
    return None

def renew_tor_circuit(control_port=9051, password=None, logger=None):
    """Use stem to signal NEWNYM (rotate circuit). Requires tor ControlPort configured and stem installed."""
    if not STEM_AVAILABLE:
        if logger:
            logger.warning("Stem not available: cannot rotate Tor circuit (install python-stem).")
        return False
    try:
        with Controller.from_port(port=control_port) as ctl:
            if password:
                ctl.authenticate(password=password)
            else:
                ctl.authenticate()  # try cookie or null auth
            ctl.signal(Signal.NEWNYM)
            if logger:
                logger.info("Requested NEWNYM (Tor circuit rotation).")
            return True
    except Exception as e:
        if logger:
            logger.exception("Failed to rotate Tor circuit: %s", e)
        return False

# ===================== HTML Processor =====================
class HTMLProcessor:
    def __init__(self):
        self.download_extensions = TorRendererConfig.DOWNLOAD_EXTENSIONS
        self.link_pattern = TorRendererConfig.LINK_PATTERN

    def extract_links_soup(self, html_text, base_url):
        parser_to_use = "lxml" if builder.builder_registry.lookup("lxml") else "html.parser"
        soup = BeautifulSoup(html_text, parser_to_use)
        results = []
        anchors = soup.find_all("a", href=True)
        for a in anchors:
            raw = a.get("href")
            if not raw:
                continue
            absu = urljoin(base_url, raw.split('#')[0].strip())
            if absu.endswith("/") and len(urlparse(absu).path) > 1:
                absu = absu.rstrip("/")
            results.append({
                "link": absu,
                "url_hash": sha256_text(absu),
                "domain": urlparse(absu).netloc,
                "anchor_text": (a.get_text(strip=True) or "")[:300]
            })
        # iframe src
        iframes = soup.find_all("iframe", src=True)
        for i in iframes:
            raw = i.get("src")
            if not raw:
                continue
            absu = urljoin(base_url, raw.split('#')[0].strip())
            if absu.endswith("/") and len(urlparse(absu).path) > 1:
                absu = absu.rstrip("/")
            results.append({
                "link": absu,
                "url_hash": sha256_text(absu),
                "domain": urlparse(absu).netloc,
                "anchor_text": "[iframe]"
            })
        return results

    def find_suspicious_links(self, html_text):
        links = self.link_pattern.findall(html_text)
        suspicious = []
        for link in links:
            lower = link.lower()
            for ext in self.download_extensions:
                if lower.endswith(ext) or f".{ext.lstrip('.')}" in lower:
                    suspicious.append(link)
                    break
        return suspicious
    
    def extract_js_variables(self, html_text, variable_names=None):
        """
        Extract JSON-like JS variables from <script> tags.
        :param html_text: raw HTML text
        :param variable_names: list of variable names to extract (optional)
        :return: dict of {var_name: JSON value or raw string}
        """
        results = {}
        parser_to_use = "lxml" if builder.builder_registry.lookup("lxml") else "html.parser"
        soup = BeautifulSoup(html_text, parser_to_use)
        scripts = soup.find_all("script")
        for script in scripts:
            content = script.string or ""
            if not content.strip():
                continue
            # If variable_names is None, extract any var assignment like: var something = {...};
            for line in content.splitlines():
                line = line.strip()
                if line.startswith("var ") or line.startswith("let ") or line.startswith("const "):
                    parts = line.split("=", 1)
                    if len(parts) != 2:
                        continue
                    var_name = parts[0].replace("var", "").replace("let", "").replace("const", "").strip()
                    if variable_names and var_name not in variable_names:
                        continue
                    value = parts[1].rstrip(";").strip()
                    results[var_name] = value
        return results



# ===================== Tor Browser Selenium Driver =====================
class TorBrowserSelenium:
    def __init__(self, logger, socks_port, user_agent=None, headless=False, tor_binary=None):
        self.logger = logger
        self.socks_port = socks_port
        self.user_agent = user_agent
        self.headless = headless
        self.tor_binary = tor_binary
        self.driver = None

    def create_driver(self):
        opts = Options()
        opts.headless = self.headless

        # If a Tor Browser binary path is supplied, use it.
        if self.tor_binary:
            opts.binary_location = self.tor_binary

        # Proxy to Tor SOCKS
        opts.set_preference("network.proxy.type", 1)
        opts.set_preference("network.proxy.socks", TorRendererConfig.TOR_SOCKS_HOST)
        opts.set_preference("network.proxy.socks_port", int(self.socks_port))
        opts.set_preference("network.proxy.socks_version", 5)
        # Important: tell Firefox to make DNS lookups via proxy (needed for .onion)
        opts.set_preference("network.proxy.socks_remote_dns", True)

        # Minimal privacy prefs; Tor Browser already does a lot, but set UA override and disable downloads
        if self.user_agent:
            opts.set_preference("general.useragent.override", self.user_agent)

        opts.set_preference("dom.webnotifications.enabled", False)
        opts.set_preference("browser.cache.disk.enable", False)
        opts.set_preference("browser.cache.memory.enable", False)
        opts.set_preference("browser.privatebrowsing.autostart", True)
        opts.set_preference("browser.download.folderList", 2)
        opts.set_preference("browser.download.useDownloadDir", False)
        opts.set_preference("browser.helperApps.neverAsk.saveToDisk", "")

        # Try to be conservative about telemetry
        opts.set_preference("toolkit.telemetry.enabled", False)
        opts.set_preference("datareporting.healthreport.uploadEnabled", False)

        try:
            # If geckodriver is in PATH, Service() without args is OK; otherwise set path explicitly here.
            service = FirefoxService()
            driver = webdriver.Firefox(service=service, options=opts)
            driver.set_page_load_timeout(TorRendererConfig.PAGE_LOAD_TIMEOUT)
            self.driver = driver
            self.logger.info("Started Firefox driver with Tor SOCKS %s:%s", TorRendererConfig.TOR_SOCKS_HOST, self.socks_port)
            return driver
        except WebDriverException as e:
            self.logger.exception("Failed to start Firefox webdriver: %s", e)
            raise

    def quit(self):
        try:
            if self.driver:
                self.driver.quit()
        except Exception:
            pass

# ===================== Loader (main) =====================
class Loader:
    def __init__(self, url , hash_value):
        self.url = url
        self.hash_value = hash_value
        self.config = TorRendererConfig
        self.logger = None
        self.html_processor = HTMLProcessor()
        self.driver_manager = None
        self.login_manager = LoginManager()


    def load_seen_hashes(self):
        seen = set()
        if not os.path.exists(self.config.GLOBAL_INDEX):
            return seen
        try:
            with open(self.config.GLOBAL_INDEX, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        if "url_hash" in obj:
                            seen.add(obj["url_hash"])
                    except Exception:
                        continue
        except Exception:
            pass
        return seen

    def append_to_index(self, record):
        os.makedirs(os.path.dirname(self.config.GLOBAL_INDEX), exist_ok=True)
        line = json.dumps(record, ensure_ascii=False)
        with open(self.config.GLOBAL_INDEX, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def human_like_scroll(self, driver):
        try:
            height = driver.execute_script("return Math.max(document.body.scrollHeight, document.documentElement.scrollHeight);")
            step = random.randint(200, 800)
            for y in range(0, height + step, step):
                driver.execute_script("window.scrollTo(0, arguments[0]);", y)
                time.sleep(random.uniform(0.2, 1.0))
        except JavascriptException:
            pass

    def simulate_mouse_move(self, driver):
        try:
            actions = ActionChains(driver)
            # move to a random coordinate inside the viewport
            actions.move_by_offset(random.randint(10, 200), random.randint(10, 200)).perform()
            time.sleep(random.uniform(0.1, 0.6))
        except Exception:
            pass

    def detect_captcha_or_block(self, rendered_html, driver=None):
        lower = (rendered_html or "").lower()
        # Simple heuristics: extend as needed
        keywords = ["captcha", "are you human", "please enable javascript", "access denied", "verify you are human", "cloudflare"]
        for k in keywords:
            if k in lower:
                return True, k
        # Optionally look for visible captcha elements via DOM
        if driver:
            try:
                # common selectors for CAPTCHA providers
                candidates = driver.execute_script("""
                    return Array.from(document.querySelectorAll('div,iframe,input'))
                      .map(n => ({tag: n.tagName.toLowerCase(), id: n.id || '', cls: n.className || '', outer: n.outerHTML.slice(0,200)}));
                """)
                for c in candidates[:40]:
                    if "recaptcha" in (c.get("id","") + " " + c.get("cls","")).lower():
                        return True, "recaptcha"
            except Exception:
                pass
        return False, None

    def js_extract_links(self, driver):
        """Extract links directly from live DOM using JS to catch dynamic hrefs."""
        try:
            js = """
            return Array.from(document.querySelectorAll('a[href]')).map(a => ({
              href: a.href,
              text: (a.innerText || '').slice(0,300)
            })).concat(Array.from(document.querySelectorAll('iframe[src]')).map(i=>({href:i.src, text:'[iframe]'})));
            """
            res = driver.execute_script(js)
            results = []
            for r in res:
                if not r.get("href"):
                    continue
                href = r.get("href")
                parsed = urljoin(self.url, href.split('#')[0].strip())
                if parsed.endswith("/") and len(urlparse(parsed).path) > 1:
                    parsed = parsed.rstrip("/")
                results.append({
                    "link": parsed,
                    "url_hash": sha256_text(parsed),
                    "domain": urlparse(parsed).netloc,
                    "anchor_text": (r.get("text") or "")[:300]
                })
            return results
        except Exception:
            return []

    def run(self):
        #parsed = urlparse(self.url)
        #host_safe = safe_filename(parsed.netloc or parsed.path or "site")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{self.hash_value}_{ts}"

        out_dir = os.path.join(self.config.OUTPUT_BASE, folder_name)
        os.makedirs(out_dir, exist_ok=True)

        log_path = os.path.join(out_dir, "render.log")
        self.logger = LoggerManager.setup_logger(log_path)
        self.logger.info("Output directory: %s", out_dir)
        self.logger.info("Target URL: %s", self.url)

        # Auto-detect open Tor SOCKS port
        socks_port = find_open_socks_port(host=self.config.TOR_SOCKS_HOST, ports=self.config.TOR_SOCKS_PORTS)
        if not socks_port:
            self.logger.error("No Tor SOCKS port found on localhost (checked %s). Make sure Tor is running.", self.config.TOR_SOCKS_PORTS)
            return 2
        self.logger.info("Using Tor SOCKS port: %s", socks_port)

        seen_hashes = self.load_seen_hashes()
        self.logger.info("Loaded %d seen link hashes from index.", len(seen_hashes))

        attempts = 0
        last_exc = None

        # Choose a UA for this run (rotate)
        ua = random.choice(self.config.USER_AGENTS)

        while attempts < self.config.RETRIES:
            attempts += 1
            self.logger.info("Attempt %d of %d", attempts, self.config.RETRIES)
            driver = None
            try:
                # Create driver using Tor Browser binary + SOCKS port + UA
                self.driver_manager = TorBrowserSelenium(self.logger, socks_port, user_agent=ua, headless=self.config.HEADLESS, tor_binary=self.config.TOR_BROWSER_BINARY)
                driver = self.driver_manager.create_driver()

                # Attempt to load page
                self.logger.info("Loading page (timeout=%ds)...", self.config.PAGE_LOAD_TIMEOUT)
                driver.get(self.url)
                # Try login if page includes login form
                #try:
                #    self.login_manager.login_if_required(driver, self.logger)
               # except Exception as e:
                 #   self.logger.warning("Login attempt failed: %s", e)

                # Simulate slight human interaction before waiting for JS to run
                time.sleep(random.uniform(0.4, 1.2))
                try:
                    self.simulate_mouse_move(driver)
                except Exception:
                    pass
                # Human-like scrolling while JS executes
                self.human_like_scroll(driver)

                # Wait a little to let heavy JS complete
                time.sleep(self.config.POST_LOAD_SLEEP + random.uniform(0, 2))

                # Try to extract dynamic links via JS (best)
                links_js = self.js_extract_links(driver)
                rendered = None
                try:
                    rendered = driver.page_source
                except Exception:
                    rendered = ""

                # Basic CAPTCHA / challenge detection
                is_blocked, reason = self.detect_captcha_or_block(rendered, driver)
                if is_blocked:
                    self.logger.warning("Potential CAPTCHA or bot-block detected: %s", reason)
                    # Save screenshot and page for manual inspection, then either pause (human-in-loop) or retry with circuit change
                    png_path = os.path.join(out_dir, f"screenshot_block_attempt{attempts}.png")
                    atomic_write_bytes(png_path, driver.get_screenshot_as_png())
                    atomic_write_text(os.path.join(out_dir, f"rendered_block_attempt{attempts}.html"), rendered)
                    # Try rotating Tor circuit before retrying (if stem available)
                    if STEM_AVAILABLE:
                        renew_tor_circuit(control_port=self.config.TOR_CONTROL_PORT, password=self.config.TOR_CONTROL_PASSWORD, logger=self.logger)
                        # Give Tor a moment
                        time.sleep(4)
                        raise Exception(f"Blocked by CAPTCHA-like content ({reason}). Rotated circuit and will retry.")
                    else:
                        raise Exception(f"Blocked by CAPTCHA-like content ({reason}). Stem not available to rotate circuit.")

                # If page is too large, save metadata only
                html_size = len(rendered.encode('utf-8')) if rendered else 0
                if html_size > self.config.MAX_HTML_BYTES:
                    self.logger.warning("Rendered HTML too large (%d bytes). Saving metadata only.", html_size)
                    page_hash = self.hash_value
                    meta = {
                        "url": self.url,
                        "fetched_timestamp": datetime.utcnow().isoformat() + "Z",
                        "page_sha256": page_hash,
                        "note": "html_too_large",
                        "html_size": html_size
                    }
                    atomic_write_text(os.path.join(out_dir, "metadata.json"), json.dumps(meta, ensure_ascii=False, indent=2))
                else:
                    # save HTML and screenshot
                    html_path = os.path.join(out_dir, "rendered.html")
                    png_path = os.path.join(out_dir, "screenshot.png")
                    atomic_write_text(html_path, rendered or "")
                    atomic_write_bytes(png_path, driver.get_screenshot_as_png())

                    # suspicious links
                    suspicious = self.html_processor.find_suspicious_links(rendered or "")
                    if suspicious:
                        atomic_write_text(os.path.join(out_dir, "suspicious_links.txt"), "\n".join(suspicious))
                        self.logger.warning("Detected %d suspicious download links.", len(suspicious))

                    # metadata
                    page_hash = self.hash_value
                    meta = {
                        "url": self.url,
                        "fetched_timestamp": datetime.utcnow().isoformat() + "Z",
                        "page_sha256": page_hash,
                        "html_size": html_size,
                        "suspicious_count": len(suspicious)
                    }
                    atomic_write_text(os.path.join(out_dir, "metadata.json"), json.dumps(meta, ensure_ascii=False, indent=2))

                    
                    # ---------------- Extract JS variables (like 'data') ----------------
                    js_vars = self.html_processor.extract_js_variables(rendered or "", variable_names=None)
                    if js_vars:
                       js_vars_path = os.path.join(out_dir, "extracted_js_variables.json")
                       atomic_write_text(js_vars_path, json.dumps(js_vars, ensure_ascii=False, indent=2))
                       self.logger.info("Extracted JS variables: %s", ", ".join(js_vars.keys()))
                    # ---------------------------------------------------------------------
                       
                       # Extract links: prefer JS-extracted list, then supplement with BeautifulSoup fallback
                    links = links_js or self.html_processor.extract_links_soup(rendered or "", self.url)
                    # Merge / dedupe by url_hash
                    uniq = {}
                    for item in links:
                        uniq[item["url_hash"]] = item
                    links = list(uniq.values())

                    extracted_path = os.path.join(out_dir, "extracted_links.json")
                    atomic_write_text(extracted_path, json.dumps(links, ensure_ascii=False, indent=2))
                    self.logger.info("Extracted %d links (JS + fallback).", len(links))

                    # Append new links to global index
                    new_count = 0
                    for item in links:
                        if item["url_hash"] in seen_hashes:
                            continue
                        rec = {
                            "seed_url": self.url,
                            "link": item["link"],
                            "url_hash": item["url_hash"],
                            "domain": item["domain"],
                            "anchor_text": item.get("anchor_text", ""),
                            "fetched_timestamp": datetime.utcnow().isoformat() + "Z"
                        }
                        self.append_to_index(rec)
                        seen_hashes.add(item["url_hash"])
                        new_count += 1
                    self.logger.info("Appended %d new links to global index.", new_count)

                # Graceful shutdown of driver
                try:
                    driver.quit()
                except Exception:
                    pass

                self.logger.info("Finished successfully.")
                self.logger.info("Sleeping %ds (rate-limit)...", self.config.RATE_LIMIT_SLEEP)
                time.sleep(self.config.RATE_LIMIT_SLEEP + random.uniform(0,2))
                return 0

            except KeyboardInterrupt:
                self.logger.info("Interrupted by user. Exiting.")
                last_exc = "KeyboardInterrupt"
                if driver:
                    try:
                        driver.quit()
                    except Exception:
                        pass
                break
            except TimeoutException as e:
                last_exc = e
                self.logger.warning("Timeout while loading page: %s", e)
            except Exception as e:
                last_exc = e
                self.logger.exception("Exception during render attempt: %s", e)
            finally:
                if driver:
                    try:
                        driver.quit()
                    except Exception:
                        pass

            # If we're going to retry, maybe rotate Tor circuit (best-effort)
            if attempts < self.config.RETRIES:
                self.logger.info("Retrying after %ds...", self.config.RETRY_DELAY)
                # Attempt NEWNYM if stem available
                try:
                    if STEM_AVAILABLE:
                        renew_tor_circuit(control_port=self.config.TOR_CONTROL_PORT, password=self.config.TOR_CONTROL_PASSWORD, logger=self.logger)
                        # Pause a bit for circuit to change
                        time.sleep(4)
                except Exception:
                    pass
                time.sleep(self.config.RETRY_DELAY + random.uniform(0,2))

        # Failed attempts -> write failure metadata
        try:
            fail_meta = {
                "url": self.url,
                "fetched_timestamp": datetime.utcnow().isoformat() + "Z",
                "status": "failed",
                "last_error": str(last_exc)
            }
            atomic_write_text(os.path.join(out_dir, "metadata.json"), json.dumps(fail_meta, ensure_ascii=False, indent=2))
        except Exception:
            pass

        return 2

# ===================== Signal Handler =====================
def _signal_handler(sig, frame):
    raise KeyboardInterrupt()

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

# ===================== Main =====================
def usage():
    print("Usage: python3 tor_crawler_torbrowser.py <url>")
    print("Example: python3 tor_crawler_torbrowser.py http://exampleonionaddress.onion")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 filename.py <site_url> <site_hash>")
        sys.exit(1)

    target_url = sys.argv[1].strip()
    target_hash = sys.argv[2].strip()

    if not (target_url.startswith("http://") or target_url.startswith("https://")):
        print("Please include scheme (http:// or https://).")
        sys.exit(1)

    load = Loader(target_url, target_hash)
    exit_code = load.run()
    sys.exit(exit_code)
