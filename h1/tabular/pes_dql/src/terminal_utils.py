"""
pes_dql — Terminal Utilities for the Pandemic Experiment Scenario (Q-Learning v2).

Provides styled text formatting and visual elements for enhanced console output.
Enables creation of formatted headers, sections, progress indicators, and data displays.

Key Features
------------
• Formatted headers and sections with ASCII borders
• Colored status indicators: success (✓), error (❌), info (ℹ)
• Progress bar visualization with percentage tracking
• Data row formatting with customizable column widths
• Box and banner elements for visual emphasis
• List item formatting with indentation levels

Main Functions
---------------
• header: Large formatted header with border
• section: Subsection title with separator
• success/error/info: Status messages with icons
• progress: Animated progress bar display
• list_item: Formatted list items with hierarchy
• data_row: Columnar data display
• box/banner: Visual container elements
"""

def header(text, width=80):
    """
    Print a large formatted header with top and bottom borders.

    Creates an emphasized section header with equal signs as borders,
    useful for marking major sections of program output.

    Parameters
    ----------
    text : str
        Header text to display
    width : int, optional
        Total width of the header including borders. Default: 80

    Returns
    -------
    None
        Prints to stdout

    Examples
    --------
    >>> header('EXPERIMENT CONFIGURATION')
    # Prints:
    # ================================================================================
    #   EXPERIMENT CONFIGURATION
    # ================================================================================
    """
    print(f"\n{'='*width}")
    print(f"  {text}")
    print(f"{'='*width}\n")

def section(text, width=80):
    """
    Print a formatted section subtitle with separator line.

    Creates a subsection title with dashes, useful for organizing
    output into logical subsections under headers.

    Parameters
    ----------
    text : str
        Section title text
    width : int, optional
        Total width of the section including borders. Default: 80

    Returns
    -------
    None
        Prints to stdout
    """
    print(f"\n{'-'*width}")
    print(f"  {text}")
    print(f"{'-'*width}\n")

def success(text, prefix="✓"):
    """
    Print a success message with checkmark prefix.

    Parameters
    ----------
    text : str
        Success message
    prefix : str, optional
        Prefix symbol. Default: ✓ (checkmark)

    Returns
    -------
    None
        Prints to stdout
    """
    print(f"{prefix} {text}")

def error(text, prefix="❌"):
    """
    Print an error message with cross mark prefix.

    Parameters
    ----------
    text : str
        Error message
    prefix : str, optional
        Prefix symbol. Default: ❌ (cross mark)

    Returns
    -------
    None
        Prints to stdout
    """
    print(f"{prefix} {text}")

def info(text, prefix="ℹ"):
    """
    Print an informational message with info symbol prefix.

    Parameters
    ----------
    text : str
        Information message
    prefix : str, optional
        Prefix symbol. Default: ℹ (info symbol)

    Returns
    -------
    None
        Prints to stdout
    """
    print(f"{prefix} {text}")

def progress(current, total, text="Progress", width=40):
    """
    Display an animated progress bar with percentage indicator.

    Prints a visual progress bar that updates in place, showing completion
    percentage and current/total counts. Automatically adds newline when complete.

    Parameters
    ----------
    current : int
        Current iteration count
    total : int
        Total iterations to complete
    text : str, optional
        Label for the progress bar. Default: "Progress"
    width : int, optional
        Width of the progress bar in characters. Default: 40

    Returns
    -------
    None
        Prints to stdout with carriage return (no newline until completion)

    Examples
    --------
    >>> for i in range(101):
    ...     progress(i, 100, "Training", width=30)
    # Displays: Training: [███...░░] 50% (50/100)
    """
    filled = int(width * current / total)
    bar = "█" * filled + "░" * (width - filled)
    pct = int(100 * current / total)
    print(f"\r{text}: [{bar}] {pct}% ({current}/{total})", end="", flush=True)
    if current == total:
        print()

def list_item(text, level=1, prefix="→"):
    """
    Print a formatted list item with optional indentation for hierarchy.

    Parameters
    ----------
    text : str
        Item text to display
    level : int, optional
        Indentation level (1-based). Default: 1 (no indent)
        Level 2 = 2 spaces, Level 3 = 4 spaces, etc.
    prefix : str, optional
        Bullet symbol. Default: → (right arrow)

    Returns
    -------
    None
        Prints to stdout

    Examples
    --------
    >>> list_item('Main item', level=1)
    # → Main item
    >>> list_item('Sub item', level=2)
    #   → Sub item
    """
    indent = "  " * (level - 1)
    print(f"{indent}{prefix} {text}")

def data_row(label, value, label_width=40, value_width=35):
    """
    Print a formatted key-value data row with aligned columns.

    Useful for displaying configuration parameters and results in
    columnar format with consistent width and alignment.

    Parameters
    ----------
    label : str
        Label/key text
    value : str or numeric
        Value to display (auto-converted to string)
    label_width : int, optional
        Width allocated for label column. Default: 40
    value_width : int, optional
        Width allocated for value column. Default: 35

    Returns
    -------
    None
        Prints to stdout

    Examples
    --------
    >>> data_row('NUM_BLOCKS', 8)
    >>> data_row('PANDEMIC_PARAMETER', 0.4)
    # Output: NUM_BLOCKS                      8 and parameter values aligned
    """
    print(f"{label:<{label_width}} {value:<{value_width}}")

def separator(width=80, char="="):
    """
    Print a horizontal separator line.

    Parameters
    ----------
    width : int, optional
        Width of separator in characters. Default: 80
    char : str, optional
        Character to use for separator. Default: "="

    Returns
    -------
    None
        Prints to stdout
    """
    print(f"{char*width}")

def banner(text, width=80, char="="):
    """
    Print a centered banner with decorative borders.

    Creates a prominent centered text with top and bottom borders,
    useful for highlighting special messages or section breaks.

    Parameters
    ----------
    text : str
        Banner text (will be centered)
    width : int, optional
        Banner width. Default: 80
    char : str, optional
        Border character. Default: "="

    Returns
    -------
    None
        Prints to stdout
    """
    _padding = (width - len(text) - 2) // 2
    print(f"{char*width}")
    print(f"{char} {text.center(width-4)} {char}")
    print(f"{char*width}\n")

def box(text, width=80):
    """
    Print text in a decorated box with borders.

    Wraps text in a box made of box-drawing characters. Handles
    multi-line text by boxing each line individually.

    Parameters
    ----------
    text : str
        Text to display (can contain newlines)
    width : int, optional
        Maximum box width. Default: 80

    Returns
    -------
    None
        Prints to stdout
    """
    lines = text.split('\n')
    max_len = max(len(line) for line in lines) if lines else 0
    box_width = min(max(max_len + 4, width), width)

    print(f"┌{'─'*(box_width-2)}┐")
    for line in lines:
        padding = box_width - 4 - len(line)
        print(f"│ {line}{' '*padding} │")
    print(f"└{'─'*(box_width-2)}┘\n")

def highlight(text, _width=80):
    """
    Print text with a highlight prefix arrow.

    Parameters
    ----------
    text : str
        Text to highlight
    width : int, optional
        Currently unused. Included for API consistency. Default: 80

    Returns
    -------
    None
        Prints to stdout
    """
    print(f"▶ {text}")
