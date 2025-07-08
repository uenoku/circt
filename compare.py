# REQUIRES: bindings_python
# RUN: %PYTHON% %s | FileCheck %s

import circt
from circt.dialects import aig, hw
from circt.ir import Context, Location, Module, InsertionPoint, IntegerType, FlatSymbolRefAttr, StringAttr
from circt.dialects.aig import LongestPathAnalysis, LongestPathCollection
import argparse
import logging
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

try:
    from IPython import embed
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("IPython not available. Interactive mode will be limited.")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Create a simple fallback for tqdm
    def tqdm(iterable, desc=None, total=None):
        if desc:
            print(f"{desc}...")
        return iterable

parser = argparse.ArgumentParser(description="Compare longest path analysis between two MLIR files")
parser.add_argument("mlir_file_old", help="Path to the old MLIR file")
parser.add_argument("mlir_file_new", help="Path to the new MLIR file")
parser.add_argument("module_name", help="Name of the module to analyze")
# Optional filtering arguments
parser.add_argument("--interesting_fanout", help="Filter paths by fanout signal name (optional)")
parser.add_argument("--interesting_fanin", help="Filter paths by fanin signal name (optional)")
# Logging arguments
parser.add_argument("--log-level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                   default='INFO', help="Set the logging level (default: INFO)")
parser.add_argument("--log-file", help="Write logs to a file instead of stdout")
# Interactive mode
parser.add_argument("--interactive", "-i", action="store_true",
                   help="Enable interactive mode for exploring paths")
# Progress bar option
parser.add_argument("--no-progress", action="store_true",
                   help="Disable progress bars (useful for scripting)")
# Parallel execution option
parser.add_argument("--parallel", action="store_true",
                   help="Run analysis on both designs in parallel (faster for large designs)")
args = parser.parse_args()

# Configure logging
log_level = getattr(logging, args.log_level.upper())
log_format = '%(asctime)s - %(levelname)s - %(message)s'

if args.log_file:
    logging.basicConfig(filename=args.log_file, level=log_level, format=log_format)
else:
    logging.basicConfig(level=log_level, format=log_format)

logger = logging.getLogger(__name__)


def load_module_from_file(path: Path, ctx: Context) -> Module:
    """This functions loads an MLIR Module from a file at the given Path."""
    logger.info(f"Loading MLIR module from file: {path}")

    try:
        with ctx:
            with open(path, "rb") as text_or_bc:
                logger.debug(f"Reading file content from {path}")
                module = Module.parse(text_or_bc.read())
                logger.info(f"Successfully parsed MLIR module from {path}")
        return module
    except Exception as e:
        logger.error(f"Failed to load module from {path}: {e}")
        raise


def filter_paths_by_interest(collection: LongestPathCollection, interesting_fanout: str = None, interesting_fanin: str = None):
    """Filter paths based on interesting fanout and/or fanin signal names."""
    logger.info(f"Filtering paths - fanout filter: '{interesting_fanout}', fanin filter: '{interesting_fanin}'")

    if not interesting_fanout and not interesting_fanin:
        logger.info("No filtering criteria specified, returning all paths")
        return collection  # No filtering needed

    logger.debug(f"Starting with {len(collection)} paths")
    filtered_paths = []

    # Create progress bar description
    filter_desc = []
    if interesting_fanout:
        filter_desc.append(f"fanout='{interesting_fanout}'")
    if interesting_fanin:
        filter_desc.append(f"fanin='{interesting_fanin}'")
    desc = f"Filtering paths by {', '.join(filter_desc)}"

    # Use tqdm for progress tracking
    disable_progress = not TQDM_AVAILABLE or args.no_progress if 'args' in globals() else not TQDM_AVAILABLE
    for i in tqdm(range(len(collection)), desc=desc, disable=disable_progress):
        path = collection[i]

        # Check fanout filter
        fanout_matches = True
        if interesting_fanout:
            fanout_matches = interesting_fanout in path.fan_out.name
            logger.debug(f"Path {i}: fanout '{path.fan_out.name}' {'matches' if fanout_matches else 'does not match'} filter '{interesting_fanout}'")

        # Check fanin filter
        fanin_matches = True
        if interesting_fanin:
            fanin_matches = interesting_fanin in path.path.fan_in.name
            logger.debug(f"Path {i}: fanin '{path.path.fan_in.name}' {'matches' if fanin_matches else 'does not match'} filter '{interesting_fanin}'")

        # Include path if both filters match (or are not specified)
        if fanout_matches and fanin_matches:
            filtered_paths.append(path)
            logger.debug(f"Path {i}: included in filtered results")
        else:
            logger.debug(f"Path {i}: excluded from filtered results")

    logger.info(f"Filtering complete: {len(filtered_paths)} paths match the criteria")
    return filtered_paths


def show_path_details(path, index=None):
    """Display detailed information about a single path."""
    prefix = f"Path {index}: " if index is not None else "Path: "
    print(f"{prefix}")
    print(f"  Fanout: {path.fan_out.name}")
    print(f"  Fanin:  {path.path.fan_in.name}")
    print(f"  Delay:  {path.delay}")
    print(f"  Root:   {path.root}")


def compare_paths(paths_old, paths_new, max_paths=10):
    """Compare paths between old and new designs."""
    print(f"\n=== Path Comparison (showing up to {max_paths} paths) ===")

    old_list = paths_old if isinstance(paths_old, list) else [paths_old[i] for i in range(min(max_paths, len(paths_old)))]
    new_list = paths_new if isinstance(paths_new, list) else [paths_new[i] for i in range(min(max_paths, len(paths_new)))]

    max_len = max(len(old_list), len(new_list))

    print(f"{'Index':<5} {'Old Design':<50} {'New Design':<50} {'Delay Diff'}")
    print("-" * 120)

    for i in range(max_len):
        old_path = old_list[i] if i < len(old_list) else None
        new_path = new_list[i] if i < len(new_list) else None

        old_str = f"{old_path.fan_out.name}->{old_path.path.fan_in.name} ({old_path.delay})" if old_path else "N/A"
        new_str = f"{new_path.fan_out.name}->{new_path.path.fan_in.name} ({new_path.delay})" if new_path else "N/A"

        if old_path and new_path:
            delay_diff = new_path.delay - old_path.delay
            diff_str = f"{delay_diff:+d}"
        else:
            diff_str = "N/A"

        print(f"{i+1:<5} {old_str:<50} {new_str:<50} {diff_str}")


def find_paths_by_signal(collection, signal_name, search_fanout=True, search_fanin=True):
    """Find paths containing a specific signal name."""
    matching_paths = []

    # Create progress bar description
    search_desc = []
    if search_fanout:
        search_desc.append("fanout")
    if search_fanin:
        search_desc.append("fanin")
    desc = f"Searching {'/'.join(search_desc)} for '{signal_name}'"

    disable_progress = not TQDM_AVAILABLE or args.no_progress if 'args' in globals() else not TQDM_AVAILABLE
    for i in tqdm(range(len(collection)), desc=desc, disable=disable_progress):
        path = collection[i]
        match = False

        if search_fanout and signal_name in path.fan_out.name:
            match = True
        if search_fanin and signal_name in path.path.fan_in.name:
            match = True

        if match:
            matching_paths.append((i, path))

    return matching_paths


def run_single_analysis(module, module_name, fanout_filter, fanin_filter, design_name):
    """Run longest path analysis on a single design."""
    thread_id = threading.current_thread().ident
    logger.info(f"[Thread {thread_id}] Starting analysis for {design_name}")

    try:
        analysis = LongestPathAnalysis(module.operation, trace_debug_points=True)
        logger.debug(f"[Thread {thread_id}] Created analysis object for {design_name}")

        collection = analysis.get_all_paths(module_name, fanout_filter, fanin_filter)
        logger.info(f"[Thread {thread_id}] Found {len(collection)} paths in {design_name} (after CAPI filtering)")

        return {
            'design_name': design_name,
            'analysis': analysis,
            'collection': collection,
            'success': True,
            'error': None
        }
    except Exception as e:
        logger.error(f"[Thread {thread_id}] Failed to analyze {design_name}: {e}")
        return {
            'design_name': design_name,
            'analysis': None,
            'collection': None,
            'success': False,
            'error': str(e)
        }



logger.info("Starting longest path analysis comparison")

with Context() as ctx, Location.unknown():
  logger.debug("Registering CIRCT dialects")
  circt.register_dialects(ctx)
  ctx.enable_multithreading(False)

  # Parse from arguments
  logger.info(f"Loading modules: old='{args.mlir_file_old}', new='{args.mlir_file_new}'")
  m_old = load_module_from_file(Path(args.mlir_file_old), ctx)
  m_new = load_module_from_file(Path(args.mlir_file_new), ctx)
  ctx.enable_multithreading(True)

  # m_old 
  attr_name = "aig.longest-path-analysis-top"
  marker = FlatSymbolRefAttr.get(args.module_name)

  m_old.operation.attributes[attr_name] = marker
  m_new.operation.attributes[attr_name] = marker

  logger.info(f"Preparing to analyze module: '{args.module_name}'")

  # Prepare filter strings for CAPI (empty string means no filter)
  fanout_filter = args.interesting_fanout if args.interesting_fanout else ""
  fanin_filter = args.interesting_fanin if args.interesting_fanin else ""

  if fanout_filter or fanin_filter:
    logger.info(f"Using CAPI filtering: fanout='{fanout_filter}', fanin='{fanin_filter}'")

  if args.parallel:
    logger.info("Running path analysis on both designs in parallel")
    start_time = time.time()

    # Run analyses in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as executor:
      # Submit both analysis tasks
      future_old = executor.submit(run_single_analysis, m_old, args.module_name,
                                   fanout_filter, fanin_filter, "old design")
      future_new = executor.submit(run_single_analysis, m_new, args.module_name,
                                   fanout_filter, fanin_filter, "new design")

      # Collect results as they complete
      results = {}
      for future in as_completed([future_old, future_new]):
        result = future.result()
        results[result['design_name']] = result

        if result['success']:
          logger.info(f"Completed analysis for {result['design_name']}")
        else:
          logger.error(f"Failed analysis for {result['design_name']}: {result['error']}")
          sys.exit(1)

    parallel_time = time.time() - start_time
    logger.info(f"Parallel analysis completed in {parallel_time:.2f} seconds")

    # Extract results
    analysis_old = results['old design']['analysis']
    analysis_new = results['new design']['analysis']
    collection_old = results['old design']['collection']
    collection_new = results['new design']['collection']

  else:
    logger.info("Running path analysis on both designs sequentially")
    start_time = time.time()

    try:
      analysis_old = LongestPathAnalysis(m_old.operation, trace_debug_points=True)
      collection_old = analysis_old.get_all_paths(args.module_name, fanout_filter, fanin_filter)
      logger.info(f"Found {len(collection_old)} paths in old design (after CAPI filtering)")
    except Exception as e:
      logger.error(f"Failed to get paths from old design: {e}")
      sys.exit(1)

    try:
      analysis_new = LongestPathAnalysis(m_new.operation, trace_debug_points=True)
      collection_new = analysis_new.get_all_paths(args.module_name, fanout_filter, fanin_filter)
      logger.info(f"Found {len(collection_new)} paths in new design (after CAPI filtering)")
    except Exception as e:
      logger.error(f"Failed to get paths from new design: {e}")
      sys.exit(1)

    sequential_time = time.time() - start_time
    logger.info(f"Sequential analysis completed in {sequential_time:.2f} seconds")

  # CHECK-LABEL:      LongestPathAnalysis created successfully!
  print("LongestPathAnalysis created successfully!")
  logger.info("Analysis completed successfully")

  # Print filtering results
  if args.interesting_fanout or args.interesting_fanin:
    logger.info("Displaying CAPI filtering results")
    print(f"Filtered paths: old={len(collection_old)}, new={len(collection_new)}")
    if args.interesting_fanout:
      print(f"Filtered by fanout containing: '{args.interesting_fanout}'")
    if args.interesting_fanin:
      print(f"Filtered by fanin containing: '{args.interesting_fanin}'")
  else:
    print(f"Total paths: old={len(collection_old)}, new={len(collection_new)}")
    logger.info("No filtering was applied")

  # Use the collections directly since filtering was done in CAPI
  paths_old = collection_old
  paths_new = collection_new

  # Compare the two analyses
  diff = collection_old.diff(collection_new)
  diff.print_summary()

  logger.info("Displaying example paths from analysis results")

  # Display some example paths to show the results
  print("\n--- Example paths from old design ---")
  logger.debug(f"Showing first 3 paths from {len(paths_old)} total paths (old design)")
  for i in range(min(3, len(paths_old))):  # Show first 3 paths
    path = paths_old[i]
    print(f"Path {i+1}: fanout={path.fan_out.name}, fanin={path.path.fan_in.name}, delay={path.delay}")

  print("\n--- Example paths from new design ---")
  logger.debug(f"Showing first 3 paths from {len(paths_new)} total paths (new design)")
  for i in range(min(3, len(paths_new))):  # Show first 3 paths
    path = paths_new[i]
    print(f"Path {i+1}: fanout={path.fan_out.name}, fanin={path.path.fan_in.name}, delay={path.delay}")

logger.info("Analysis comparison completed successfully")

# Interactive mode
if args.interactive:
  logger.info("Starting interactive mode")
  if not IPYTHON_AVAILABLE:
    print("Warning: IPython not available. Install with: pip install ipython")
    print("Falling back to basic interactive mode...")

    # Basic interactive loop
    while True:
      try:
        print("\n=== Interactive Path Analysis ===")
        print("Available commands:")
        print("  'old' - Show paths from old design")
        print("  'new' - Show paths from new design")
        print("  'filter <fanout> <fanin>' - Apply new filters")
        print("  'stats' - Show statistics")
        print("  'help' - Show this help")
        print("  'quit' - Exit interactive mode")

        cmd = input("\nEnter command: ").strip().lower()

        if cmd == 'quit':
          break
        elif cmd == 'old':
          print(f"\n--- Paths from old design ({len(paths_old)} total) ---")
          for i in range(min(10, len(paths_old))):
            path = paths_old[i] if isinstance(paths_old, list) else paths_old[i]
            print(f"Path {i+1}: fanout={path.fan_out.name}, fanin={path.path.fan_in.name}, delay={path.delay}")
        elif cmd == 'new':
          print(f"\n--- Paths from new design ({len(paths_new)} total) ---")
          for i in range(min(10, len(paths_new))):
            path = paths_new[i] if isinstance(paths_new, list) else paths_new[i]
            print(f"Path {i+1}: fanout={path.fan_out.name}, fanin={path.path.fan_in.name}, delay={path.delay}")
        elif cmd == 'stats':
          print(f"\nStatistics:")
          print(f"  Origin/cal paths: old={len(collection_old)}, new={len(collection_new)}")
          if isinstance(paths_old, list):
            print(f"  Filtered paths: old={len(paths_old)}, new={len(paths_new)}")
        elif cmd == 'help':
          continue  # Will show help again
        else:
          print("Unknown command. Type 'help' for available commands.")

      except KeyboardInterrupt:
        print("\nExiting interactive mode...")
        break
      except Exception as e:
        print(f"Error: {e}")

  else:
    # IPython interactive mode
    print("\n=== Starting IPython Interactive Session ===")
    print("Available variables:")
    print("  collection_old, collection_new - Path collections (already filtered by CAPI)")
    print("  paths_old, paths_new - Same as collections (for compatibility)")
    print("  analysis_old, analysis_new - Analysis objects")
    print("  m_old, m_new - MLIR modules")
    print("\nAvailable helper functions:")
    print("  filter_paths_by_interest(collection, fanout, fanin) - Python-based filter (slower)")
    print("  show_path_details(path, index) - Show detailed path info")
    print("  compare_paths(paths_old, paths_new, max_paths) - Compare path collections")
    print("  find_paths_by_signal(collection, signal_name) - Find paths with signal")
    print("\nExample usage:")
    print("  len(collection_old)  # Number of paths")
    print("  collection_old[0]    # First path")
    print("  show_path_details(paths_old[0], 1)  # Show details of first path")
    print("  compare_paths(paths_old, paths_new, 5)  # Compare first 5 paths")
    print("  clk_paths = find_paths_by_signal(collection_old, 'clk')  # Find clock paths")
    print("  # Note: For efficient filtering, use --interesting_fanout/fanin arguments")
    print("  # The collections are already filtered by CAPI if those arguments were used")

    # Start IPython session with all variables available
    embed()