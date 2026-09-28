#!/usr/bin/env python3
"""Download and OCR Aruba's latest monthly hotel-performance table."""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
from datetime import date
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import pytesseract
import requests
import numpy as np
from bs4 import BeautifulSoup
from PIL import Image, ImageOps


PAGE_URL = "https://tourismanalytics.com/aruba-statistics.html"
OUTPUT_FILE = Path(
    os.environ.get("ARUBA_OUTPUT_FILE", str(Path.cwd() / "aruba_hotel_history.xlsx"))
).expanduser()
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "YTD"]
BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/140.0.0.0 Safari/537.36"
)


def fetch_bytes(url: str) -> bytes:
    """Fetch a URL, falling back to macOS curl when requests is blocked."""
    try:
        response = requests.get(
            url,
            timeout=30,
            headers={
                "User-Agent": BROWSER_UA,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        response.raise_for_status()
        return response.content
    except requests.RequestException as request_error:
        try:
            result = subprocess.run(
                [
                    "curl", "--location", "--fail", "--silent", "--show-error",
                    "--retry", "3", "--user-agent", BROWSER_UA, url,
                ],
                check=True,
                capture_output=True,
            )
            return result.stdout
        except (FileNotFoundError, subprocess.CalledProcessError) as curl_error:
            raise RuntimeError(
                f"Could not download {url} with requests or curl. "
                f"requests error: {request_error}; curl error: {curl_error}"
            ) from curl_error


def latest_table_image() -> tuple[str, str]:
    try:
        page_html = fetch_bytes(PAGE_URL)
        soup = BeautifulSoup(page_html, "html.parser")
        for tag in soup.find_all("img"):
            alt = tag.get("alt", "")
            if re.search(r"Aruba.*Hotel Performance.*20\d{2}", alt, re.I):
                src = tag.get("src") or tag.get("data-src")
                if src:
                    return urljoin(PAGE_URL, src), alt
    except (requests.RequestException, RuntimeError):
        pass

    # Some hosting networks receive a stripped anti-bot page. The publisher's
    # monthly image filenames are predictable, so probe recent months directly.
    today = date.today()
    for offset in range(6):
        month_index = today.year * 12 + today.month - 1 - offset
        year, zero_based_month = divmod(month_index, 12)
        month_number = zero_based_month + 1
        month_name = [
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december",
        ][month_number - 1]
        for year_label in (str(year), str(year)[-2:]):
            candidate = (
                "https://tourismanalytics.weebly.com/uploads/1/2/0/4/120443739/"
                f"ahata-{month_name}-{year_label}_orig.png"
            )
            try:
                content = fetch_bytes(candidate)
                if content.startswith(b"\x89PNG\r\n\x1a\n"):
                    return candidate, (
                        f"Aruba Hotel Performance YTD {year} {month_name.title()}"
                    )
            except RuntimeError:
                continue

    previous = Path("public/latest.json")
    if previous.exists():
        saved = json.loads(previous.read_text(encoding="utf-8"))
        if saved.get("source_image") and saved.get("report"):
            return saved["source_image"], saved["report"]
    raise RuntimeError("The current Aruba hotel-performance image was not found.")


def parse_metric(
    image: Image.Image,
    metric: str,
    row_boundaries: list[int],
    row_labels: list[str],
    percentage: bool = False,
) -> pd.DataFrame:
    """Read cells individually so the table's grid does not confuse OCR."""
    gray = np.asarray(ImageOps.grayscale(image))
    y_start, y_end = row_boundaries[0], row_boundaries[-1]
    dark_counts = (gray[y_start:y_end + 1, :] < 80).sum(axis=0)
    dark_x = np.where(dark_counts > (y_end - y_start) * 0.65)[0]
    groups: list[list[int]] = []
    for x in dark_x:
        if not groups or x > groups[-1][-1] + 1:
            groups.append([int(x)])
        else:
            groups[-1].append(int(x))
    vertical_lines = [round(sum(group) / len(group)) for group in groups]
    if len(vertical_lines) < 5:
        raise RuntimeError(
            f"Could not detect the table columns for {metric}: {vertical_lines}"
        )
    columns = {
        1: (vertical_lines[1], vertical_lines[2]),
        2: (vertical_lines[2], vertical_lines[3]),
    }
    rows = []

    for row_number, month in enumerate(row_labels):
        y1 = row_boundaries[row_number] + 3
        y2 = row_boundaries[row_number + 1] - 3
        for year_column, (x1, x2) in columns.items():
            cell = image.crop((int(x1) + 3, int(y1), int(x2) - 3, int(y2)))
            cell = ImageOps.grayscale(
                cell.resize((cell.width * 3, cell.height * 3))
            )
            text = pytesseract.image_to_string(cell, config="--psm 7").strip()
            clean = re.sub(r"[$,%\s]", "", text).replace("O", "0").replace("o", "0")
            number = re.search(r"\d+(?:\.\d+)?", clean)
            if not number:
                raise RuntimeError(
                    f"OCR could not read {metric}, {month}, year column "
                    f"{year_column}. Cell text was {text!r}."
                )
            value = float(number.group())
            if percentage:
                value /= 100
            rows.append({"Month": month, "YearColumn": year_column, metric: value})

    result = pd.DataFrame(rows)
    return result.drop_duplicates(["Month", "YearColumn"])


def horizontal_grid_lines(image: Image.Image) -> list[int]:
    """Locate full-width horizontal table borders in either image template."""
    gray = np.asarray(ImageOps.grayscale(image))
    dark_counts = (gray < 80).sum(axis=1)
    dark_y = np.where(dark_counts > image.width * 0.70)[0]
    groups: list[list[int]] = []
    for y in dark_y:
        if not groups or y > groups[-1][-1] + 1:
            groups.append([int(y)])
        else:
            groups[-1].append(int(y))
    lines = [round(sum(group) / len(group)) for group in groups]
    if len(lines) < 20:
        raise RuntimeError(f"Could not detect enough table rows: {lines}")
    return lines


def main() -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    image_url, report_name = latest_table_image()
    print(f"Downloading {image_url}")
    image_content = fetch_bytes(image_url)
    try:
        image = Image.open(io.BytesIO(image_content)).convert("RGB")
    except Image.UnidentifiedImageError:
        asset_url = image_url.replace(
            "https://tourismanalytics.com/uploads/",
            "https://tourismanalytics.weebly.com/uploads/",
        )
        image_content = fetch_bytes(asset_url)
        image = Image.open(io.BytesIO(image_content)).convert("RGB")
        image_url = asset_url

    year_match = re.search(r"20\d{2}", report_name)
    report_year = int(year_match.group()) if year_match else date.today().year
    full_months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]
    # The page's alt text can lag behind the image itself (August 2026 was
    # still labelled July), so treat the asset filename as authoritative.
    report_month_match = re.search(
        rf"ahata-({'|'.join(full_months)})-", image_url, re.I
    ) or re.search(rf"\b({'|'.join(full_months)})\b", report_name, re.I)
    if not report_month_match:
        raise RuntimeError(f"Could not determine the report month from {report_name!r}.")
    report_month_number = [m.lower() for m in full_months].index(
        report_month_match.group(1).lower()
    ) + 1
    report_name = (
        f"Aruba Hotel Performance YTD {report_year} "
        f"{full_months[report_month_number - 1]}"
    )
    row_labels = MONTHS[:report_month_number] + ["YTD"]

    grid_lines = horizontal_grid_lines(image)
    rows_per_metric = len(row_labels)
    occupancy_index = 3
    adr_index = occupancy_index + rows_per_metric + 3
    revpar_index = adr_index + rows_per_metric + 3
    occupancy_rows = grid_lines[occupancy_index:occupancy_index + rows_per_metric + 1]
    adr_rows = grid_lines[adr_index:adr_index + rows_per_metric + 1]
    revpar_rows = grid_lines[revpar_index:revpar_index + rows_per_metric + 1]

    occupancy = parse_metric(image, "Occupancy", occupancy_rows, row_labels, True)
    adr = parse_metric(image, "ADR", adr_rows, row_labels)
    revpar = parse_metric(image, "RevPAR", revpar_rows, row_labels)

    data = occupancy.merge(adr, on=["Month", "YearColumn"], how="outer")
    data = data.merge(revpar, on=["Month", "YearColumn"], how="outer")

    data["Year"] = data["YearColumn"].map({1: report_year, 2: report_year - 1})
    data["MonthNumber"] = data["Month"].map(
        {month: number for number, month in enumerate(MONTHS[:12], start=1)}
    )
    data["ReportDate"] = date.today().isoformat()
    data["SourceReport"] = report_name
    data["SourceImage"] = image_url
    data = data[["ReportDate", "SourceReport", "Year", "MonthNumber", "Month",
                 "Occupancy", "ADR", "RevPAR", "SourceImage"]]

    monthly = data[data["Month"] != "YTD"].copy()
    ytd = data[data["Month"] == "YTD"].copy()

    if monthly[["Occupancy", "ADR", "RevPAR"]].isna().any().any():
        raise RuntimeError("OCR missed a value; the workbook was not changed.")
    if not monthly["Occupancy"].between(0, 1).all():
        raise RuntimeError("Occupancy validation failed; the workbook was not changed.")

    gap = (monthly["RevPAR"] - monthly["ADR"] * monthly["Occupancy"]).abs()
    if (gap > 1.25).any():
        bad = monthly.loc[gap > 1.25, ["Year", "Month"]]
        labels = ", ".join(f"{row.Year} {row.Month}" for row in bad.itertuples())
        raise RuntimeError(f"RevPAR validation failed for {labels}; workbook unchanged.")

    if OUTPUT_FILE.exists():
        try:
            old = pd.read_excel(OUTPUT_FILE, sheet_name="Monthly_History")
            monthly = pd.concat([old, monthly], ignore_index=True)
        except (ValueError, KeyError):
            pass

    monthly = (
        monthly.sort_values(["Year", "MonthNumber", "ReportDate"], ascending=[True, True, False])
        .drop_duplicates(["Year", "MonthNumber"], keep="first")
        .sort_values(["Year", "MonthNumber"])
    )

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        monthly.to_excel(writer, sheet_name="Monthly_History", index=False)
        ytd.to_excel(writer, sheet_name="Current_YTD", index=False)

        for sheet_name in ("Monthly_History", "Current_YTD"):
            sheet = writer.book[sheet_name]
            sheet.freeze_panes = "A2"
            for cell in sheet["F"][1:]:
                cell.number_format = "0.0%"
            for column in ("G", "H"):
                for cell in sheet[column][1:]:
                    cell.number_format = "$#,##0.00"
            for column_cells in sheet.columns:
                width = min(max(len(str(cell.value or "")) for cell in column_cells) + 2, 60)
                sheet.column_dimensions[column_cells[0].column_letter].width = width

    csv_file = OUTPUT_FILE.with_suffix(".csv")
    json_file = OUTPUT_FILE.with_name("latest.json")
    monthly.to_csv(csv_file, index=False)
    public_payload = {
        "updated": date.today().isoformat(),
        "source_page": PAGE_URL,
        "source_image": image_url,
        "report": report_name,
        "monthly": json.loads(
            monthly[["Year", "MonthNumber", "Month", "Occupancy", "ADR", "RevPAR"]]
            .to_json(orient="records")
        ),
        "ytd": json.loads(
            ytd[["Year", "Month", "Occupancy", "ADR", "RevPAR"]]
            .to_json(orient="records")
        ),
    }
    json_file.write_text(json.dumps(public_payload, indent=2), encoding="utf-8")

    print(f"Updated {OUTPUT_FILE.resolve()}")
    print(f"Monthly rows: {len(monthly)}; current YTD rows: {len(ytd)}")


if __name__ == "__main__":
    main()
