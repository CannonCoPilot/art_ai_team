# Variant Finder Refactor — Comprehensive Requirements Document

**Version:** 2.0  
**Date:** 2025-10-02  
**Author:** Copilot Analysis (Gemini Pro 2.5 + GPT-5 synthesis)  
**Target Audience:** Autonomous AI coding agent team (Roo Code framework)

---

## 1. Project Goal

Deliver a **robust, simple, well-tested CLI tool** that:

1. **Parses an AssayName** from an input PDF file
2. **Connects to a remote Linux server** via SSH/SFTP
3. **Discovers analyses** matching the AssayName by searching `analysisSamples.json` files
4. **Constructs VCF file paths** for each sample by transforming `annotatedVariants` paths
5. **Scans each VCF file** for presence/absence of variants from an Excel input list
6. **Generates a DataFrame** with one row per sample and one column per variant (yes/no)
7. **Exports results** to timestamped CSV and Excel files with summary statistics

---

## 2. Current State Analysis

### 2.1 Existing Architecture
- **Modular Python package** with separate concerns: CLI, parsing, remote access, data building
- **CLI framework:** Currently uses `click`; must migrate to `argparse`
- **Core modules exist** but have incomplete/incorrect implementations

### 2.2 Critical Gaps (Development Blockers)
1. **No actual remote connectivity** — `remote_finder.py` uses local `os.walk()` instead of SSH/SFTP
2. **Wrong PDF parsing target** — searching for RNA config instead of hotspotFile pattern
3. **Missing VCF scanning logic** — no FILTER==PASS check, no Hotspot ID matching to VCF ID column
4. **Broken data flow** — `api.py` doesn't chain modules correctly per requirements
5. **No control sample filtering** — all samples included instead of excluding controls
6. **No LibPrep ID mapping** — cannot associate VCF files with sample identifiers

### 2.3 Why Development Is Stalling
- **Mock/placeholder code** prevents actual functionality testing
- **Incorrect assumptions** about data structures and file formats
- **Missing integration** between modules causes circular development
- **No end-to-end validation** against real fixture files

---

## 3. Functional Requirements

### FR1: Input Processing — PDF Parsing

#### FR1.1 Extract AssayName from PDF
**Requirement:**  
Parse the PDF text to extract the AssayName string found between specific markers.

**Pattern to Match:**
```
Post processing script config json:/share/apps/IR/AssayImport/{AssayName}/actorPostScriptConfig/tvcscript.json
```

**Example Input (from PDF):**
```
Post processing script config json:/share/apps/IR/AssayImport/Ts-OncomineDxTarget-AssayDefFile-JPv310_3.10.0/actorPostScriptConfig/tvcscript.json
```

**Expected Output:**
```
Ts-OncomineDxTarget-AssayDefFile-JPv310_3.10.0
```

**Implementation Guidance:**
```python
import re
from pathlib import Path
from typing import Optional
import pypdf

def parse_assay_name(pdf_path: Path) -> str:
    """
    Extract AssayName from PDF by finding text between specific markers.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        The extracted AssayName string
        
    Raises:
        ValueError: If AssayName cannot be found
        FileNotFoundError: If PDF file doesn't exist
        
    Example:
        >>> parse_assay_name(Path("audit.pdf"))
        'Ts-OncomineDxTarget-AssayDefFile-JPv310_3.10.0'
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    # Extract text from PDF
    text = ""
    with open(pdf_path, 'rb') as f:
        reader = pypdf.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    
    # Remove all whitespace to handle line breaks
    cleaned_text = re.sub(r'\s+', '', text)
    
    # Define markers
    start_marker = "Postprocessingscriptconfigjson:/share/apps/IR/AssayImport/"
    end_marker = "/actorPostScriptConfig/tvcscript.json"
    
    # Build regex pattern
    pattern = re.escape(start_marker) + r'([\w\-\.]+)' + re.escape(end_marker)
    
    match = re.search(pattern, cleaned_text, re.IGNORECASE)
    
    if not match:
        # Show user a snippet for manual verification
        snippet_start = max(0, cleaned_text.find("AssayImport") - 50)
        snippet_end = cleaned_text.find("AssayImport") + 200
        snippet = cleaned_text[snippet_start:snippet_end]
        
        raise ValueError(
            f"Could not extract AssayName from PDF.\n"
            f"Text snippet around 'AssayImport':\n{snippet}\n\n"
            f"Please verify the PDF contains the expected pattern."
        )
    
    assay_name = match.group(1)
    return assay_name
```

**Acceptance Criteria:**
- ✅ Successfully extracts `Ts-OncomineDxTarget-AssayDefFile-JPv310_3.10.0` from `Audit_Details-1758832307744.pdf`
- ✅ Handles whitespace and line breaks in PDF text
- ✅ Raises `ValueError` with helpful message if pattern not found
- ✅ Provides text snippet for manual verification on failure

---

### FR2: Input Processing — Excel Parsing

#### FR2.1 Extract Hotspot IDs from Excel
**Requirement:**  
Read an Excel file and extract a list of variant IDs (Hotspot IDs) from a specified column.

**Implementation Guidance:**
```python
from pathlib import Path
from typing import List
import pandas as pd

def parse_variant_ids(
    excel_path: Path, 
    column_name: str = "Hotspot ID"
) -> List[str]:
    """
    Parse variant IDs from Excel file, preserving order and deduplicating.
    
    Args:
        excel_path: Path to Excel file
        column_name: Name of column containing variant IDs
        
    Returns:
        Ordered list of unique variant IDs
        
    Raises:
        FileNotFoundError: If Excel file doesn't exist
        ValueError: If column not found or no IDs present
        
    Example:
        >>> parse_variant_ids(Path("variants.xlsx"))
        ['COSM12345', 'COSM67890', 'COSM11111']
    """
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")
    
    try:
        df = pd.read_excel(excel_path)
    except Exception as e:
        raise ValueError(f"Could not read Excel file: {e}")
    
    if column_name not in df.columns:
        available = ", ".join(df.columns)
        raise ValueError(
            f"Column '{column_name}' not found in Excel file.\n"
            f"Available columns: {available}"
        )
    
    # Extract column, drop nulls, convert to string
    variant_ids = df[column_name].dropna().astype(str).tolist()
    
    if not variant_ids:
        raise ValueError(f"No variant IDs found in column '{column_name}'")
    
    # Deduplicate while preserving order (dict.fromkeys trick)
    unique_ids = list(dict.fromkeys(variant_ids))
    
    return unique_ids
```

**Acceptance Criteria:**
- ✅ Extracts IDs from `ERBB2_EGFR_Variant_List.xlsx`
- ✅ Preserves original order of variants
- ✅ Removes duplicates
- ✅ Validates non-empty result
- ✅ Provides helpful error if column missing

---

### FR3: Remote Server Connectivity

#### FR3.1 Establish SSH/SFTP Connection
**Requirement:**  
Connect to remote Linux server using password authentication via SSH/SFTP.

**Implementation Guidance:**
```python
import paramiko
from contextlib import contextmanager
from typing import Generator
import logging

logger = logging.getLogger(__name__)

@contextmanager
def sftp_connection(
    server: str,
    username: str,
    password: str,
    timeout: int = 30
) -> Generator[paramiko.SFTPClient, None, None]:
    """
    Context manager for SFTP connection with automatic cleanup.
    
    Args:
        server: Server hostname or IP
        username: SSH username
        password: SSH password
        timeout: Connection timeout in seconds
        
    Yields:
        Active SFTP client
        
    Raises:
        paramiko.AuthenticationException: If credentials invalid
        paramiko.SSHException: If connection fails
        
    Example:
        >>> with sftp_connection("server.com", "user", "pass") as sftp:
        ...     files = sftp.listdir("/data")
    """
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    sftp = None
    
    try:
        logger.info(f"Connecting to {server} as {username}")
        ssh.connect(
            hostname=server,
            username=username,
            password=password,
            timeout=timeout,
            look_for_keys=False,
            allow_agent=False
        )
        sftp = ssh.open_sftp()
        logger.info("SFTP connection established")
        yield sftp
        
    except paramiko.AuthenticationException:
        logger.error("Authentication failed - check credentials")
        raise
    except paramiko.SSHException as e:
        logger.error(f"SSH connection error: {e}")
        raise
    finally:
        if sftp:
            sftp.close()
        ssh.close()
        logger.info("SFTP connection closed")
```

**Acceptance Criteria:**
- ✅ Successfully connects with valid credentials
- ✅ Raises clear exception on auth failure
- ✅ Automatically closes connection on exit
- ✅ Supports timeout configuration
- ✅ Logs connection lifecycle

---

### FR4: Remote Analysis Discovery

#### FR4.1 Recursive Search for analysisSamples.json
**Requirement:**  
Recursively search remote directory for `analysisSamples.json` files matching pattern `Auto*/Auto*/analysisSamples.json`.

#### FR4.2 Filter by AssayName in hotspotFile
**Requirement:**  
For each JSON file found, parse it and include only those where the `hotspotFile` value contains the AssayName as a substring.

**Implementation Guidance:**
```python
import json
from pathlib import Path, PurePosixPath
from typing import List, Dict, Any
import paramiko

def find_matching_analyses(
    sftp: paramiko.SFTPClient,
    root_path: str,
    assay_name: str
) -> List[Dict[str, Any]]:
    """
    Recursively find analysisSamples.json files matching assay.
    
    Args:
        sftp: Active SFTP client
        root_path: Root directory to search (e.g., /data/IR/data/analysis_output/)
        assay_name: AssayName to match in hotspotFile
        
    Returns:
        List of dicts with keys: 'json_path', 'sampleGroupIds', 'sampleExternalIds', 'annotatedVariants'
        
    Example:
        >>> analyses = find_matching_analyses(sftp, "/data/", "Ts-OncomineDx...")
        >>> len(analyses)
        5
    """
    matching_analyses = []
    
    def recursive_search(current_path: str):
        """Recursively traverse directories."""
        try:
            for entry in sftp.listdir_attr(current_path):
                entry_path = str(PurePosixPath(current_path) / entry.filename)
                
                # If directory, recurse
                if entry.st_mode & 0o040000:  # Directory bit
                    recursive_search(entry_path)
                    
                # If analysisSamples.json, check it
                elif entry.filename == "analysisSamples.json":
                    try:
                        with sftp.file(entry_path, 'r') as f:
                            data = json.load(f)
                        
                        # Check if hotspotFile contains assay_name
                        hotspot_file = data.get("hotspotFile", "")
                        if assay_name in hotspot_file:
                            matching_analyses.append({
                                'json_path': entry_path,
                                'sampleGroupIds': data.get('sampleGroupIds', []),
                                'sampleExternalIds': data.get('sampleExternalIds', []),
                                'annotatedVariants': data.get('annotatedVariants', [])
                            })
                            logger.info(f"Found matching analysis: {entry_path}")
                    
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Could not parse {entry_path}: {e}")
                    
        except IOError as e:
            logger.warning(f"Could not access {current_path}: {e}")
    
    recursive_search(root_path)
    return matching_analyses
```

**Acceptance Criteria:**
- ✅ Recursively searches all subdirectories
- ✅ Correctly filters by AssayName substring in hotspotFile
- ✅ Handles malformed JSON gracefully
- ✅ Logs progress and warnings
- ✅ Returns list of matching analyses with required fields

---

### FR5: Sample Processing

#### FR5.1 Extract and Deduplicate Sample IDs
**Requirement:**  
From all matching analyses, extract `sampleGroupIds` and `sampleExternalIds`, deduplicate while preserving order.

#### FR5.2 Filter Control Samples
**Requirement:**  
Exclude samples with IDs containing control keywords (case-insensitive).

**Control Keywords:**
- PositiveControl
- NtcControl
- Positive_Control
- NTC
- NTControl

**Implementation Guidance:**
```python
from typing import List, Dict, Set, Tuple

CONTROL_KEYWORDS = [
    "positivecontrol",
    "ntccontrol", 
    "positive_control",
    "ntc",
    "ntcontrol"
]

def is_control_sample(sample_id: str) -> bool:
    """Check if sample ID indicates a control sample."""
    sample_lower = sample_id.lower()
    return any(keyword in sample_lower for keyword in CONTROL_KEYWORDS)

def extract_samples(
    analyses: List[Dict[str, Any]]
) -> List[Dict[str, str]]:
    """
    Extract unique samples from analyses, excluding controls.
    
    Args:
        analyses: List of analysis dicts from find_matching_analyses()
        
    Returns:
        List of dicts with keys: 'sampleGroupId', 'sampleExternalId'
        Order preserved, duplicates removed, controls excluded
        
    Example:
        >>> samples = extract_samples(analyses)
        >>> samples[0]
        {'sampleGroupId': 'LibPrep3240', 'sampleExternalId': 'Sample001'}
    """
    seen_samples: Set[Tuple[str, str]] = set()
    samples: List[Dict[str, str]] = []
    
    for analysis in analyses:
        group_ids = analysis['sampleGroupIds']
        external_ids = analysis['sampleExternalIds']
        
        # Zip together, ensuring lengths match
        if len(group_ids) != len(external_ids):
            logger.warning(
                f"Mismatched sample ID lengths in {analysis['json_path']}: "
                f"{len(group_ids)} group IDs vs {len(external_ids)} external IDs"
            )
            continue
        
        for group_id, external_id in zip(group_ids, external_ids):
            # Skip controls
            if is_control_sample(group_id) or is_control_sample(external_id):
                logger.debug(f"Filtering control sample: {group_id} / {external_id}")
                continue
            
            # Deduplicate
            sample_tuple = (group_id, external_id)
            if sample_tuple not in seen_samples:
                seen_samples.add(sample_tuple)
                samples.append({
                    'sampleGroupId': group_id,
                    'sampleExternalId': external_id
                })
    
    return samples
```

**Acceptance Criteria:**
- ✅ Deduplicates samples across multiple analyses
- ✅ Preserves original discovery order
- ✅ Excludes all control keywords (case-insensitive)
- ✅ Handles mismatched ID array lengths gracefully
- ✅ Logs filtered controls at DEBUG level

---

### FR6: VCF Path Construction

#### FR6.1 Transform annotatedVariants Paths
**Requirement:**  
For each sample, construct the VCF file path by:
1. Taking `annotatedVariants` path from JSON
2. Replacing `AnnotatorActor-00` → `VariantCallerActor-00`
3. Replacing `annotated_variants.vcf` → `VariantOutput.filtered.vcf`

#### FR6.2 Extract LibPrep ID and Map to Sample
**Requirement:**  
Extract the LibPrep ID from the path (e.g., `LibPrep3240`) and map it to the corresponding `sampleGroupId`.

**Implementation Guidance:**
```python
import re
from typing import List, Dict, Optional
from pathlib import PurePosixPath

def transform_vcf_path(annotated_path: str) -> str:
    """
    Transform annotatedVariants path to VariantOutput.filtered.vcf path.
    
    Args:
        annotated_path: Original path from analysisSamples.json
        
    Returns:
        Transformed VCF path
        
    Example:
        >>> path = "/data/outputs/AnnotatorActor-00/LibPrep3240/annotated_variants.vcf"
        >>> transform_vcf_path(path)
        '/data/outputs/VariantCallerActor-00/LibPrep3240/VariantOutput.filtered.vcf'
    """
    path = annotated_path.replace("AnnotatorActor-00", "VariantCallerActor-00")
    path = path.replace("annotated_variants.vcf", "VariantOutput.filtered.vcf")
    return path

def extract_libprep_id(path: str) -> Optional[str]:
    """
    Extract LibPrep ID from path.
    
    Args:
        path: VCF file path
        
    Returns:
        LibPrep ID (e.g., 'LibPrep3240') or None if not found
        
    Example:
        >>> extract_libprep_id("/data/outputs/VariantCallerActor-00/LibPrep3240/file.vcf")
        'LibPrep3240'
    """
    match = re.search(r'(LibPrep\d+)', path)
    return match.group(1) if match else None

def build_sample_vcf_mapping(
    samples: List[Dict[str, str]],
    analyses: List[Dict[str, Any]]
) -> List[Dict[str, str]]:
    """
    Build complete mapping of samples to VCF paths.
    
    Args:
        samples: List from extract_samples()
        analyses: List from find_matching_analyses()
        
    Returns:
        List of dicts with keys: 'sampleGroupId', 'sampleExternalId', 'vcf_path'
        
    Example:
        >>> mapping = build_sample_vcf_mapping(samples, analyses)
        >>> mapping[0]
        {
            'sampleGroupId': 'LibPrep3240',
            'sampleExternalId': 'Sample001',
            'vcf_path': '/data/.../VariantCallerActor-00/LibPrep3240/VariantOutput.filtered.vcf'
        }
    """
    # Build LibPrep -> VCF path index
    libprep_to_vcf: Dict[str, str] = {}
    
    for analysis in analyses:
        for annotated_path in analysis['annotatedVariants']:
            vcf_path = transform_vcf_path(annotated_path)
            libprep_id = extract_libprep_id(vcf_path)
            
            if libprep_id:
                libprep_to_vcf[libprep_id] = vcf_path
    
    # Map samples to VCF paths
    sample_vcf_mapping = []
    
    for sample in samples:
        group_id = sample['sampleGroupId']
        
        # Try to find matching VCF path
        vcf_path = libprep_to_vcf.get(group_id)
        
        if vcf_path:
            sample_vcf_mapping.append({
                'sampleGroupId': group_id,
                'sampleExternalId': sample['sampleExternalId'],
                'vcf_path': vcf_path
            })
        else:
            logger.warning(f"No VCF path found for sample {group_id}")
    
    return sample_vcf_mapping
```

**Acceptance Criteria:**
- ✅ Correctly transforms both path components
- ✅ Extracts LibPrep ID using regex
- ✅ Maps samples to VCF paths via LibPrep ID
- ✅ Logs warning for unmapped samples
- ✅ Handles missing LibPrep IDs gracefully

---

### FR7: VCF Variant Scanning

#### FR7.1 Scan VCF for Hotspot IDs with PASS Filter
**Requirement:**  
For each VCF file and each Hotspot ID:
1. Read VCF line by line (stream, don't load entire file)
2. Skip header lines (starting with `#`)
3. Split line by tabs into columns
4. Check if FILTER column (index 6) contains "PASS"
5. Check if ID column (index 2) exactly matches Hotspot ID
6. Record "yes" on first match, "no" if never found

**VCF Column Structure (0-indexed):**
- Column 0: CHROM
- Column 1: POS
- Column 2: **ID** ← Match Hotspot ID here
- Column 3: REF
- Column 4: ALT
- Column 5: QUAL
- Column 6: **FILTER** ← Must contain "PASS"
- Column 7+: INFO, FORMAT, samples...

**Implementation Guidance:**
```python
from typing import Set, Dict
import paramiko

def scan_vcf_for_variants(
    sftp: paramiko.SFTPClient,
    vcf_path: str,
    hotspot_ids: Set[str]
) -> Dict[str, bool]:
    """
    Scan remote VCF file for presence of Hotspot IDs in PASS-filtered lines.
    
    Args:
        sftp: Active SFTP client
        vcf_path: Remote path to VCF file
        hotspot_ids: Set of Hotspot IDs to search for
        
    Returns:
        Dict mapping each Hotspot ID to True (found) or False (not found)
        
    Example:
        >>> results = scan_vcf_for_variants(sftp, "/data/sample.vcf", {"COSM12345", "COSM67890"})
        >>> results
        {'COSM12345': True, 'COSM67890': False}
    """
    # Initialize all variants as not found
    results = {hid: False for hid in hotspot_ids}
    
    try:
        with sftp.file(vcf_path, 'r') as f:
            for line_bytes in f:
                line = line_bytes.decode('utf-8').strip()
                
                # Skip header lines
                if line.startswith('#'):
                    continue
                
                # Split into columns
                fields = line.split('\t')
                
                # Validate column count
                if len(fields) < 7:
                    continue
                
                vcf_id = fields[2]      # Column 3 (0-indexed 2)
                vcf_filter = fields[6]  # Column 7 (0-indexed 6)
                
                # Only process PASS lines
                if 'PASS' not in vcf_filter:
                    continue
                
                # Check if this ID matches any Hotspot ID
                if vcf_id in hotspot_ids:
                    results[vcf_id] = True
                    
    except FileNotFoundError:
        logger.warning(f"VCF file not found: {vcf_path}")
    except Exception as e:
        logger.error(f"Error scanning VCF {vcf_path}: {e}")
    
    return results
```

**Acceptance Criteria:**
- ✅ Streams file without loading entire VCF into memory
- ✅ Correctly identifies FILTER column and checks for "PASS"
- ✅ Exactly matches Hotspot ID to VCF ID column
- ✅ Handles missing VCF files gracefully
- ✅ Returns dict with all Hotspot IDs (found or not found)
- ✅ Stops searching for a variant after first PASS match (optimization)

---

### FR8: Output Generation

#### FR8.1 Build DataFrame
**Requirement:**  
Create a pandas DataFrame with:
- Columns: `path`, `sampleGroupId`, `sampleExternalId`, then one column per Hotspot ID
- One row per sample
- Variant columns contain "yes" or "no"

#### FR8.2 Export to CSV and Excel
**Requirement:**  
Save DataFrame to:
- CSV: `{assay_name}_variant_presence_{timestamp}.csv`
- Excel: `{assay_name}_variant_presence_{timestamp}.xlsx`

#### FR8.3 Generate Summary Statistics
**Requirement:**  
Calculate and log:
- Total analyses scanned
- Total samples processed
- Total variants searched
- Variants per sample (min/max/mean)
- Samples containing each variant

**Implementation Guidance:**
```python
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Set

def build_variant_dataframe(
    sample_mapping: List[Dict[str, str]],
    hotspot_ids: List[str],
    scan_results: Dict[str, Dict[str, bool]]
) -> pd.DataFrame:
    """
    Build DataFrame with variant presence/absence.
    
    Args:
        sample_mapping: List from build_sample_vcf_mapping()
        hotspot_ids: Ordered list from parse_variant_ids()
        scan_results: Dict mapping sampleGroupId -> {hotspot_id: bool}
        
    Returns:
        DataFrame with columns: path, sampleGroupId, sampleExternalId, {variants}
        
    Example:
        >>> df = build_variant_dataframe(mapping, ids, results)
        >>> df.columns.tolist()
        ['path', 'sampleGroupId', 'sampleExternalId', 'COSM12345', 'COSM67890']
    """
    rows = []
    
    for sample in sample_mapping:
        group_id = sample['sampleGroupId']
        row = {
            'path': sample['vcf_path'],
            'sampleGroupId': group_id,
            'sampleExternalId': sample['sampleExternalId']
        }
        
        # Add variant columns
        for hotspot_id in hotspot_ids:
            is_present = scan_results.get(group_id, {}).get(hotspot_id, False)
            row[hotspot_id] = "yes" if is_present else "no"
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df

def export_results(
    df: pd.DataFrame,
    assay_name: str,
    output_dir: Path
) -> Dict[str, Path]:
    """
    Export DataFrame to CSV and Excel with timestamped filenames.
    
    Args:
        df: DataFrame from build_variant_dataframe()
        assay_name: AssayName for filename
        output_dir: Directory to save files
        
    Returns:
        Dict with keys 'csv' and 'xlsx' mapping to file paths
        
    Example:
        >>> paths = export_results(df, "Ts-OncomineDx...", Path("/output"))
        >>> paths['csv']
        PosixPath('/output/Ts-OncomineDx_variant_presence_20251002_031115.csv')
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{assay_name}_variant_presence_{timestamp}"
    
    csv_path = output_dir / f"{base_name}.csv"
    xlsx_path = output_dir / f"{base_name}.xlsx"
    
    # Export CSV
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV: {csv_path}")
    
    # Export Excel
    df.to_excel(xlsx_path, index=False, engine='openpyxl')
    logger.info(f"Saved Excel: {xlsx_path}")
    
    return {'csv': csv_path, 'xlsx': xlsx_path}

def print_summary(
    df: pd.DataFrame,
    num_analyses: int,
    hotspot_ids: List[str]
):
    """
    Print summary statistics.
    
    Args:
        df: DataFrame from build_variant_dataframe()
        num_analyses: Number of analyses scanned
        hotspot_ids: List of Hotspot IDs
    """
    num_samples = len(df)
    num_variants = len(hotspot_ids)
    
    # Count variants per sample
    variant_cols = [col for col in df.columns if col not in ['path', 'sampleGroupId', 'sampleExternalId']]
    variants_per_sample = (df[variant_cols] == "yes").sum(axis=1)
    
    # Count samples per variant
    samples_per_variant = (df[variant_cols] == "yes").sum(axis=0)
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Analyses scanned:        {num_analyses}")
    print(f"Samples processed:       {num_samples}")
    print(f"Variants searched:       {num_variants}")
    print(f"\nVariants per sample:")
    print(f"  Min:                   {variants_per_sample.min()}")
    print(f"  Max:                   {variants_per_sample.max()}")
    print(f"  Mean:                  {variants_per_sample.mean():.2f}")
    print(f"\nSamples containing each variant:")
    for variant_id in hotspot_ids:
        count = samples_per_variant.get(variant_id, 0)
        print(f"  {variant_id:20s} {count:3d} samples")
    print("="*60 + "\n")
```

**Acceptance Criteria:**
- ✅ DataFrame has exactly the specified columns in order
- ✅ Variant columns contain only "yes" or "no"
- ✅ Files saved with correct naming pattern including timestamp
- ✅ Both CSV and Excel exports work
- ✅ Summary statistics are accurate and formatted clearly

---

### FR9: CLI Interface

#### FR9.1 Replace click with argparse
**Requirement:**  
Migrate from `click` to `argparse` to support both interactive prompts and flag-based operation.

#### FR9.2 Support Interactive and Non-Interactive Modes
**Requirement:**  
- Interactive mode: Prompt for missing arguments
- Non-interactive mode (`--non-interactive`): All required args must be provided as flags

**Implementation Guidance:**
```python
import argparse
import getpass
import sys
from pathlib import Path
from typing import Optional

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Namespace with all arguments
    """
    parser = argparse.ArgumentParser(
        description="Variant Finder - Scan VCF files for variant presence/absence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (prompts for missing inputs)
  variant_finder  
  
  # Non-interactive mode (all flags required)
  variant_finder --pdf audit.pdf --excel variants.xlsx \
                 --server 10.0.0.1 --user admin --password secret \
                 --non-interactive  
  
  # With custom root and output directory
  variant_finder --pdf audit.pdf --excel variants.xlsx \
                 --server 10.0.0.1 --user admin --password secret \
                 --root /custom/path --out /output/dir --non-interactive
        """
    )
    
    parser.add_argument(
        '--pdf',
        type=Path,
        help='Path to PDF file containing AssayName'
    )
    
    parser.add_argument(
        '--excel',
        type=Path,
        help='Path to Excel file containing Hotspot IDs'
    )
    
    parser.add_argument(
        '--server',
        help='Remote server hostname or IP address'
    )
    
    parser.add_argument(
        '--user',
        help='SSH username'
    )
    
    parser.add_argument(
        '--password',
        help='SSH password (prompt recommended for security)'
    )
    
    parser.add_argument(
        '--root',
        default='/serverdata/dxdata/IR/data/analysis_output/',
        help='Root directory on remote server to search (default: %(default)s)'
    )
    
    parser.add_argument(
        '--out',
        type=Path,
        default=Path.cwd() / 'output',
        help='Output directory for results (default: %(default)s)'
    )
    
    parser.add_argument(
        '--non-interactive',
        action='store_true',
        help='Run in non-interactive mode (no prompts)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose (DEBUG) logging'
    )
    
    return parser.parse_args()

def prompt_for_missing_args(args: argparse.Namespace) -> argparse.Namespace:
    """
    Prompt user for any missing required arguments.
    
    Args:
        args: Parsed arguments from argparse
        
    Returns:
        Updated arguments with user-provided values
    """
    if args.non_interactive:
        # Validate all required args present
        missing = []
        if not args.pdf:
            missing.append('--pdf')
        if not args.excel:
            missing.append('--excel')
        if not args.server:
            missing.append('--server')
        if not args.user:
            missing.append('--user')
        if not args.password:
            missing.append('--password')
        
        if missing:
            print(f"ERROR: Non-interactive mode requires all arguments.", file=sys.stderr)
            print(f"Missing: {', '.join(missing)}", file=sys.stderr)
            sys.exit(1)
    
    else:
        # Interactive prompts
        if not args.pdf:
            pdf_input = input("Path to PDF file: ").strip()
            args.pdf = Path(pdf_input) if pdf_input else None
        
        if not args.excel:
            excel_input = input("Path to Excel file: ").strip()
            args.excel = Path(excel_input) if excel_input else None
        
        if not args.server:
            args.server = input("Remote server address: ").strip()
        
        if not args.user:
            args.user = input("SSH username: ").strip()
        
        if not args.password:
            args.password = getpass.getpass("SSH password: ")
    
    return args

def cli_main():
    """
    Main CLI entry point.
    """
    args = parse_arguments()
    args = prompt_for_missing_args(args)
    
    # Validate inputs
    if not args.pdf or not args.pdf.exists():
        print(f"ERROR: PDF file not found: {args.pdf}", file=sys.stderr)
        sys.exit(1)
    
    if not args.excel or not args.excel.exists():
        print(f"ERROR: Excel file not found: {args.excel}", file=sys.stderr)
        sys.exit(1)
    
    # Setup logging
    from variant_finder.logging_setup import setup_logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(args.out / 'variant_finder.log', log_level)
    
    # Run main workflow
    try:
        from variant_finder.api import find_variants
        result_paths = find_variants(args)
        
        print("\nSUCCESS! Results saved to:")
        print(f"  CSV:   {result_paths['csv']}")
        print(f"  Excel: {result_paths['xlsx']}")
        
    except Exception as e:
        logger.exception("Fatal error during execution")
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    cli_main()
```

**Acceptance Criteria:**
- ✅ `--help` displays all options with examples
- ✅ Interactive mode prompts for missing inputs
- ✅ Non-interactive mode validates all required flags present
- ✅ Password prompt uses `getpass` (hidden input)
- ✅ Clear error messages on invalid inputs
- ✅ Exits with code 1 on errors, 0 on success

---

### FR10: Logging and Error Handling

#### FR10.1 Implement Rotating Log Handler
**Requirement:**  
Use `RotatingFileHandler` with 5MB max size and 3 backups.

#### FR10.2 Log Levels
**Requirement:**  
- INFO: Progress and major steps
- WARNING: Missing files, skipped samples
- ERROR: Connection failures, unrecoverable errors
- DEBUG: Detailed traces (when `--verbose` flag used)

**Implementation Guidance:**
```python
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

def setup_logging(log_file: Path, level: int = logging.INFO) -> logging.Logger:
    """
    Configure logging with rotating file handler and console output.
    
    Args:
        log_file: Path to log file
        level: Logging level (logging.INFO, logging.DEBUG, etc.)
        
    Returns:
        Configured logger instance
        
    Example:
        >>> logger = setup_logging(Path("output/app.log"), logging.DEBUG)
        >>> logger.info("Application started")
    """
    # Create logger
    logger = logging.getLogger("variant_finder")
    logger.setLevel(level)
    logger.propagate = False
    
    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler (rotating)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3,
        encoding='utf-8'
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    logger.info("Logging initialized")
    return logger
```

**Acceptance Criteria:**
- ✅ Log file rotates at 5MB
- ✅ Keeps 3 backup files
- ✅ Logs to both file and console
- ✅ Respects verbose flag for DEBUG level
- ✅ Timestamps in consistent format

---

## 4. Non-Functional Requirements

### NFR1: Performance
- Process up to 500 samples in under 15 minutes
- Stream VCF files (don't load entire file into memory)
- Reuse SSH connection for all operations

### NFR2: Reliability
- Handle network interruptions gracefully
- Validate all file paths before processing
- Log all warnings and errors comprehensively

### NFR3: Security
- Never log passwords
- Use `getpass` for password prompts (hidden input)
- Close SSH connections properly on exit or error

### NFR4: Maintainability
- All functions have type hints
- All functions have docstrings with examples
- Code coverage >80%
- Follow PEP 8 style guidelines

### NFR5: Usability
- Clear progress indicators for long operations
- Informative error messages with suggested fixes
- Summary statistics at completion

---

## 5. Success Criteria & Acceptance Tests

### Acceptance Test Suite

#### AT1: PDF Parsing
```python
def test_parse_assay_name_success():
    """Test parsing AssayName from fixture PDF."""
    assay = parse_assay_name(Path("gene_lists/Audit_Details-1758832307744.pdf"))
    assert assay == "Ts-OncomineDxTarget-AssayDefFile-JPv310_3.10.0"

def test_parse_assay_name_not_found():
    """Test error handling when AssayName not in PDF."""
    with pytest.raises(ValueError, match="Could not extract AssayName"):
        parse_assay_name(Path("invalid.pdf"))
```

#### AT2: Excel Parsing
```python
def test_parse_variant_ids_success():
    """Test extracting Hotspot IDs from fixture Excel."""
    ids = parse_variant_ids(Path("gene_lists/ERBB2_EGFR_Variant_List.xlsx"))
    assert isinstance(ids, list)
    assert len(ids) > 0
    assert len(ids) == len(set(ids))  # No duplicates

def test_parse_variant_ids_missing_column():
    """Test error when column not found."""
    with pytest.raises(ValueError, match="Column.*not found"):
        parse_variant_ids(Path("gene_lists/ERBB2_EGFR_Variant_List.xlsx"), column="InvalidColumn")
```

#### AT3: Control Filtering
```python
@pytest.mark.parametrize("sample_id,expected", [
    ("PositiveControl", True),
    ("NtcControl", True),
    ("Sample123", False),
    ("LibPrep3240", False),
    ("positive_control_01", True),
    ("NTC", True),
])
def test_is_control_sample(sample_id, expected):
    """Test control sample detection."""
    assert is_control_sample(sample_id) == expected
```

#### AT4: Path Transformation
```python
def test_transform_vcf_path():
    """Test VCF path transformation."""
    input_path = "/data/outputs/AnnotatorActor-00/LibPrep3240/annotated_variants.vcf"
    expected = "/data/outputs/VariantCallerActor-00/LibPrep3240/VariantOutput.filtered.vcf"
    assert transform_vcf_path(input_path) == expected
    
def test_extract_libprep_id():
    """Test LibPrep ID extraction."""
    path = "/data/outputs/VariantCallerActor-00/LibPrep3240/file.vcf"
    assert extract_libprep_id(path) == "LibPrep3240"
        
    assert extract_libprep_id("/no/libprep/here") is None
```

#### AT5: VCF Scanning
```python
def test_scan_vcf_for_variants(tmp_path):
    """Test VCF scanning with PASS filter."""
    # Create test VCF
    vcf_content = """##fileformat=VCFv4.2
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO
chr1\t12345\tCOSM12345\tA\tT\t100\tPASS\tDP=50
chr2\t67890\tCOSM67890\tG\tC\t50\tLowQual\tDP=10
chr3\t11111\tCOSM11111\tC\tA\t200\tPASS\tDP=100
"""
    vcf_file = tmp_path / "test.vcf"
    vcf_file.write_text(vcf_content)
    
    # Scan for variants
    hotspot_ids = {"COSM12345", "COSM67890", "COSM99999"}
    
    # Mock SFTP for local file
    # (in real test, use mock SFTP client)
    results = scan_vcf_for_variants_local(vcf_file, hotspot_ids)
    
    assert results["COSM12345"] is True  # Found with PASS
    assert results["COSM67890"] is False  # Found but not PASS
    assert results["COSM99999"] is False  # Not found
```

#### AT6: DataFrame Building
```python
def test_build_variant_dataframe():
    """Test DataFrame construction."""
    sample_mapping = [
        {
            'sampleGroupId': 'LibPrep3240',
            'sampleExternalId': 'Sample001',
            'vcf_path': '/data/sample1.vcf'
        }
    ]
    
    hotspot_ids = ['COSM12345', 'COSM67890']
    
    scan_results = {
        'LibPrep3240': {
            'COSM12345': True,
            'COSM67890': False
        }
    }
    
    df = build_variant_dataframe(sample_mapping, hotspot_ids, scan_results)
    
    assert list(df.columns) == ['path', 'sampleGroupId', 'sampleExternalId', 'COSM12345', 'COSM67890']
    assert df.loc[0, 'COSM12345'] == "yes"
    assert df.loc[0, 'COSM67890'] == "no"
```

#### AT7: CLI Arguments
```python
def test_cli_non_interactive_missing_args(capsys):
    """Test non-interactive mode with missing arguments."""
    with pytest.raises(SystemExit) as exc_info:
        sys.argv = ['variant_finder', '--non-interactive']
        cli_main()
    
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Missing:" in captured.err
```

#### AT8: End-to-End Integration
```python
@pytest.mark.integration
def test_end_to_end_with_fixtures(tmp_path, mock_sftp):
    """Test complete workflow with fixture files."""
    # Setup
    pdf_path = Path("gene_lists/Audit_Details-1758832307744.pdf")
    excel_path = Path("gene_lists/ERBB2_EGFR_Variant_List.xlsx")
    output_dir = tmp_path / "output"
    
    # Mock SFTP to return local fixtures
    # Configure mock to return analysisSamples.json and VCF files
    
    # Run
    args = argparse.Namespace(
        pdf=pdf_path,
        excel=excel_path,
        server="mock.server",
        user="test",
        password="test",
        root="/data/",
        out=output_dir,
        verbose=False
    )
    
    result_paths = find_variants(args)
    
    # Verify
    assert result_paths['csv'].exists()
    assert result_paths['xlsx'].exists()
    
    df = pd.read_csv(result_paths['csv'])
    assert 'sampleGroupId' in df.columns
    assert 'sampleExternalId' in df.columns
    assert len(df) > 0
```

---

## 6. Guardrails for Autonomous AI Team

### Mandatory Requirements Validation
Before proceeding to the next phase in `roadmap.md`, **ALL** of the following must be true:

1. **All unit tests pass** for modules completed in current phase
2. **Code coverage ≥80%** for critical modules (pdf_parser, remote_finder, vcf_utils, dataframe_builder)
3. **Integration test passes** with fixture files
4. **All acceptance criteria in this document are met** for completed modules
5. **No TODO or placeholder code** remains in completed modules
6. **All functions have type hints and docstrings** with examples
7. **PEP 8 compliance** verified with `flake8`
8. **Logging works** and produces readable output

### Progress Tracking
- **Daily checkpoint**: Review `todo.md` and mark completed items
- **Phase gate**: Before moving to next milestone in `roadmap.md`, verify all requirements
- **Continuous validation**: Run acceptance tests after each module implementation

### Escalation Criteria
If any of the following occur, **STOP** and escalate to Orchestrator for review:

1. Any requirement in this document is ambiguous or contradicts code structure
2. Any module fails tests after 3 implementation attempts
3. Integration between modules fails unexpectedly
4. Performance is >2x worse than NFR targets
5. Fixture files don't match expected formats

---

## 7. Dependencies

### Required Python Packages
```txt
pandas>=2.0.0
openpyxl>=3.1.0
paramiko>=3.0.0
pypdf>=3.0.0
pytest>=7.4.0
pytest-cov>=4.1.0
flake8>=6.0.0
```

### Fixture Files (for testing)
- `gene_lists/Audit_Details-1758832307744.pdf` — Sample PDF with AssayName
- `gene_lists/ERBB2_EGFR_Variant_List.xlsx` — Sample Excel with Hotspot IDs
- `gene_lists/analysisSamples.json` — Sample JSON structure
- `gene_lists/VariantOutput.filtered.vcf` — Sample VCF file

---

## 8. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-09-28 | Initial requirements from Gemini Pro 2.5 analysis |
| 2.0 | 2025-10-02 | Enhanced with GPT-5 improvements, added code examples, expanded acceptance tests |

---

**End of Requirements Document**

This document serves as the contract for all deliverables. No code should be merged that doesn't meet these requirements. All changes to requirements must be approved by the Orchestrator and documented in this version history.
