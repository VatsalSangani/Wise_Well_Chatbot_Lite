#!/usr/bin/env python3
"""
PubMed FTP downloader (baseline + updatefiles)
- Downloads .xml.gz and .md5
- Verifies MD5
- Supports selecting specific file numbers
- Resumes partial downloads if a .part file exists
- Retries with backoff
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import socket
import sys
import time
from dataclasses import dataclass
from ftplib import FTP, error_perm
from pathlib import Path
from typing import Iterable, Optional, Tuple, List


NCBI_FTP_HOST = "ftp.ncbi.nlm.nih.gov"
PUBMED_BASELINE_DIR = "/pubmed/baseline"
PUBMED_UPDATE_DIR = "/pubmed/updatefiles"


@dataclass(frozen=True)
class FtpTarget:
    remote_dir: str
    name: str  # label for logging


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _md5_of_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _parse_md5_file(md5_path: Path) -> Optional[str]:
    """
    NCBI .md5 files are typically like:
      <md5sum>  <filename>
    We'll extract the first 32-hex token.
    """
    txt = md5_path.read_text(errors="ignore")
    m = re.search(r"\b([a-fA-F0-9]{32})\b", txt)
    return m.group(1).lower() if m else None


def _should_download_xml(filename: str) -> bool:
    return filename.endswith(".xml.gz") and filename.startswith("pubmed") and "n" in filename


def _extract_pubmed_file_number(filename: str) -> Optional[int]:
    """
    Example: pubmed25n0001.xml.gz -> 1
    """
    m = re.match(r"pubmed\d{2}n(\d{4})\.xml\.gz$", filename)
    if not m:
        return None
    return int(m.group(1))


def _list_remote_files(ftp: FTP, remote_dir: str) -> List[str]:
    ftp.cwd(remote_dir)
    return ftp.nlst()


def _download_with_resume(
    ftp: FTP,
    remote_filename: str,
    local_path: Path,
    retries: int = 5,
    backoff_base: float = 1.5,
) -> None:
    """
    Download remote_filename into local_path.
    Uses a .part file for resume.
    """
    part_path = local_path.with_suffix(local_path.suffix + ".part")
    attempt = 0

    while True:
        try:
            ftp.voidcmd("TYPE I")  # switch to binary mode (IMAGE) so SIZE works
            remote_size = ftp.size(remote_filename)
            if remote_size is None:
                raise RuntimeError(f"Could not get remote size for {remote_filename}")

            # If already fully present, skip
            if local_path.exists() and local_path.stat().st_size == remote_size:
                return

            # If a partial exists, resume from there
            offset = 0
            if part_path.exists():
                offset = part_path.stat().st_size
                if offset > remote_size:
                    # corrupted partial
                    part_path.unlink()

            with part_path.open("ab") as f:
                if offset > 0:
                    ftp.sendcmd(f"TYPE I")
                    ftp.retrbinary(f"RETR {remote_filename}", f.write, blocksize=1024 * 256, rest=offset)
                else:
                    ftp.retrbinary(f"RETR {remote_filename}", f.write, blocksize=1024 * 256)

            # Validate size after download
            if part_path.stat().st_size != remote_size:
                raise RuntimeError(
                    f"Size mismatch after download for {remote_filename}: "
                    f"got {part_path.stat().st_size}, expected {remote_size}"
                )

            part_path.replace(local_path)
            return

        except (socket.timeout, ConnectionResetError, EOFError, error_perm, OSError, RuntimeError) as e:
            attempt += 1
            if attempt > retries:
                raise RuntimeError(f"Failed downloading {remote_filename} after {retries} retries: {e}") from e

            sleep_s = backoff_base ** attempt
            print(f"[warn] Download error ({remote_filename}): {e}. Retrying in {sleep_s:.1f}s...", file=sys.stderr)
            time.sleep(sleep_s)


def _verify_md5(xml_path: Path, md5_path: Path) -> Tuple[bool, str, str]:
    expected = _parse_md5_file(md5_path)
    if not expected:
        return False, "", ""
    actual = _md5_of_file(xml_path)
    return actual.lower() == expected.lower(), expected.lower(), actual.lower()


def download_pubmed(
    email_password: str,
    out_dir: Path,
    target: FtpTarget,
    file_nums: Optional[Iterable[int]] = None,
    max_files: Optional[int] = None,
    verify: bool = True,
) -> None:
    _safe_mkdir(out_dir)

    with FTP(NCBI_FTP_HOST, timeout=60) as ftp:
        ftp.login(user="anonymous", passwd=email_password)
        files = _list_remote_files(ftp, target.remote_dir)

        # Filter xml files
        xml_files = [f for f in files if _should_download_xml(f)]
        xml_files.sort()

        # Apply file number filter
        if file_nums is not None:
            wanted = set(int(n) for n in file_nums)
            xml_files = [f for f in xml_files if (_extract_pubmed_file_number(f) in wanted)]

        # Apply max_files cap
        if max_files is not None:
            xml_files = xml_files[: max_files]

        print(f"[info] {target.name}: {len(xml_files)} XML files selected")

        for i, xml_name in enumerate(xml_files, start=1):
            md5_name = xml_name + ".md5"
            local_xml = out_dir / xml_name
            local_md5 = out_dir / md5_name

            # Download MD5 first (small)
            ftp.cwd(target.remote_dir)
            _download_with_resume(ftp, md5_name, local_md5)

            # If already present and verified, skip
            if verify and local_xml.exists():
                ok, exp, act = _verify_md5(local_xml, local_md5)
                if ok:
                    print(f"[ok] ({i}/{len(xml_files)}) verified, skipping: {xml_name}")
                    continue
                else:
                    print(f"[warn] MD5 mismatch, re-downloading: {xml_name} (expected {exp}, got {act})")
                    local_xml.unlink(missing_ok=True)

            # Download XML
            print(f"[dl] ({i}/{len(xml_files)}) {xml_name}")
            _download_with_resume(ftp, xml_name, local_xml)

            # Verify MD5 after download
            if verify:
                ok, exp, act = _verify_md5(local_xml, local_md5)
                if not ok:
                    local_xml.unlink(missing_ok=True)
                    raise RuntimeError(f"MD5 verification failed for {xml_name}: expected {exp}, got {act}")
                print(f"[ok] verified: {xml_name}")


def _parse_range(rng: str) -> List[int]:
    """
    Parse '1-10,15,20-22' into [1..10,15,20..22]
    """
    out: List[int] = []
    for part in rng.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a_i, b_i = int(a), int(b)
            if b_i < a_i:
                a_i, b_i = b_i, a_i
            out.extend(range(a_i, b_i + 1))
        else:
            out.append(int(part))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Download PubMed baseline/update XML files from NCBI FTP.")
    ap.add_argument("--email", required=True, help="Use a real email; NCBI requests it as the FTP password.")
    ap.add_argument("--out", required=True, help="Output directory.")
    ap.add_argument("--source", choices=["baseline", "updatefiles"], default="baseline")
    ap.add_argument("--range", default=None, help="File number ranges like '1-10,25,100-120' (uses pubmedYYnNNNN).")
    ap.add_argument("--max-files", type=int, default=None, help="Cap number of files downloaded (after filtering).")
    ap.add_argument("--no-verify", action="store_true", help="Skip MD5 verification (not recommended).")
    args = ap.parse_args()

    target = FtpTarget(
        remote_dir=PUBMED_BASELINE_DIR if args.source == "baseline" else PUBMED_UPDATE_DIR,
        name=args.source,
    )

    nums = _parse_range(args.range) if args.range else None
    out_dir = Path(args.out).expanduser().resolve()

    download_pubmed(
        email_password=args.email,
        out_dir=out_dir,
        target=target,
        file_nums=nums,
        max_files=args.max_files,
        verify=not args.no_verify,
    )

    print("[done] Downloads complete.")


if __name__ == "__main__":
    main()
