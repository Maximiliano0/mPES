"""
Logging Utilities for pes_trf (Pandemic Experiment Scenario — Transformer Deep Q-Network)

Provides centralized console logging with simultaneous file and terminal output.
Enables dual-stream logging where messages are printed to both the console and
a timestamped log file, with automatic handling of ANSI color codes.

Key Features
------------
• Singleton pattern for log file management - ensures single file handle per session
• Dual output (terminal + file) - messages appear in both streams simultaneously
• Timestamp logging - all file entries include UTC timestamps
• Color handling - ANSI colors displayed in terminal, stripped from file output
• TensorFlow support - automatic conversion of TensorFlow types to native Python types

Main Components
----------------
• create_ConsoleLog_filehandle_singleton: Initialize or get log file handle
• close_consolelog_filehandle: Gracefully close the log file
• tee: Print to both terminal and log file with timestamps
• strip_colour: Remove ANSI escape sequences from strings
• _convert_tensorflow_types: Convert TensorFlow tensors to Python types
"""

##########################
##  Imports externos    ##
##########################
import os
import datetime

##########################
##  Imports internos    ##
##########################
from .. import ANSI, OUTPUT_FILE_PREFIX, OUTPUTS_PATH, VERBOSE


##################################################
## Module variables requiring initialization    ##
##################################################
ConsoleLog_filehandle = None


def create_ConsoleLog_filehandle_singleton(SubjectId: str):
    """
    Initialize or retrieve the singleton log file handle for console output.

    Creates a file handle for logging all console output to a timestamped file in
    the outputs directory. Uses singleton pattern to ensure only one file handle
    exists per session. Subsequent calls return the existing handle.

    Parameters
    ----------
    SubjectId : str
        Unique identifier for the subject/session, used in log filename
        Format: PES_TRF_log_{SubjectId}.txt

    Returns
    -------
    file
        File handle opened in write mode with UTF-8 encoding

    Raises
    ------
    AssertionError
        If SubjectId is None

    Notes
    -----
    • This function implements the singleton pattern
    • The module-global ConsoleLog_filehandle is initialized on first call
    • To reset the singleton, close the current handle and set ConsoleLog_filehandle = None
    • File is created in OUTPUTS_PATH directory with UTC timestamp capability

    Examples
    --------
    >>> handle = create_ConsoleLog_filehandle_singleton('2026-02-09_TRF_AGENT')
    >>> # Creates: outputs/PES_TRF_log_2026-02-09_TRF_AGENT.txt
    """

    assert SubjectId is not None

    global ConsoleLog_filehandle

    if ConsoleLog_filehandle is None:
        ConsoleLog_filename = os.path.join(OUTPUTS_PATH, f'{OUTPUT_FILE_PREFIX}log_{SubjectId}.txt')
        ConsoleLog_filehandle = open(ConsoleLog_filename, 'w', encoding='utf-8')

    return ConsoleLog_filehandle


def close_consolelog_filehandle():
    """
    Close the singleton console log file handle.

    Gracefully closes the log file associated with the console logging system.
    Should be called during experiment cleanup before program termination.

    Raises
    ------
    AssertionError
        If ConsoleLog_filehandle has not been initialized
        (create_ConsoleLog_filehandle_singleton must be called first)

    Notes
    -----
    • Always call this before program termination to ensure file is properly closed
    • After calling, create_ConsoleLog_filehandle_singleton can be called again
      to start a new log file for a new session

    Examples
    --------
    >>> handle = create_ConsoleLog_filehandle_singleton('session_id')
    >>> # ... experiment runs ...
    >>> close_consolelog_filehandle()  # Clean shutdown
    """
    global ConsoleLog_filehandle
    assert ConsoleLog_filehandle is not None, "ConsoleLog_filehandle provided was None. Aborting `close` operation."
    ConsoleLog_filehandle.close()


def _convert_tensorflow_types(value):
    """
    Convert TensorFlow tensor objects to native Python types.

    Utility function for handling TensorFlow variables and tensors that may
    appear in logged output. Automatically extracts scalar values or converts
    array tensors to NumPy arrays for string representation.

    Parameters
    ----------
    value : any
        Value potentially containing TensorFlow objects

    Returns
    -------
    any
        Converted value (native Python type) or original value if not TensorFlow
        - Scalar tensors → Python scalars
        - Array tensors → NumPy arrays
        - Other types → unchanged

    Examples
    --------
    >>> import tensorflow as tf
    >>> tensor = tf.constant(42)
    >>> result = _convert_tensorflow_types(tensor)
    >>> type(result)  # Returns native Python int
    <class 'numpy.int32'>
    """
    if hasattr(value, 'numpy'):  # TensorFlow Variable or Tensor
        return value.numpy().item() if value.numpy().ndim == 0 else value.numpy()
    return value


def tee(*Strings, **kwargs):
    """
    Dual-stream logging: print to terminal AND write timestamped output to log file.

    Simultaneously outputs a message to both the console (with ANSI colors) and
    the log file (with UTC timestamp, colors stripped). Provides synchronized
    logging across both outputs. Useful for capturing all console messages in
    a persistent log while maintaining visual formatting on screen.

    Parameters
    ----------
    *Strings : str or convertible
        Variable number of values to log. Non-string values are converted to strings.
        TensorFlow types are automatically converted to native Python types.
    **kwargs : dict, optional
        Keyword arguments passed to print() (e.g., sep, end, etc.)
        Note: 'file' and 'flush' arguments are removed/overridden

    Behavior
    --------
    • Terminal output: Messages printed in ANSI Blue (if VERBOSE=True)
    • File output: Messages prefixed with UTC timestamp, no ANSI colors
    • Both outputs are flushed immediately
    • TensorFlow tensors automatically converted before logging

    Raises
    ------
    AssertionError
        If ConsoleLog_filehandle has not been initialized

    Examples
    --------
    >>> create_ConsoleLog_filehandle_singleton('session_1')
    >>> tee('Starting experiment...')
    # Terminal: ANSI_BLUE + 'Starting experiment...' + ANSI_RESET
    # File: [2026-02-09 14:23:45.123456+00:00] : Starting experiment...

    >>> tee('Iteration', 42, 'complete')
    # Multiple arguments joined with spaces, same dual output

    Notes
    -----
    • Does not output anything if VERBOSE=False (only logs to file)
    • Colors controlled by global ANSI configuration
    • Safe for repeated calls; timestamp added automatically each time
    """

    global ConsoleLog_filehandle
    assert ConsoleLog_filehandle is not None, "ConsoleLog_filehandle has not been initialised in module"

    if 'file' in kwargs:
        kwargs.pop('file')
    if 'flush' in kwargs:
        kwargs.pop('flush')

    # Convert TensorFlow types to native Python types
    Strings = [_convert_tensorflow_types(Str) for Str in Strings]
    Strings = [ANSI.BLUE + str(Str) + ANSI.RESET for Str in Strings]

    ColorlessStrings = []
    for Str in Strings:
        ColorlessStrings.append(strip_colour(Str))

    if VERBOSE:
        print(*Strings, flush=True, **kwargs)
    print('[', datetime.datetime.now(tz=datetime.timezone.utc), ']:', *
          ColorlessStrings, file=ConsoleLog_filehandle, flush=True, **kwargs)


def strip_colour(Str):
    """
    Remove all ANSI color codes from a string.

    Removes ANSI escape sequences (color, style, formatting) from a string,
    leaving only the plain text. Useful for preparing colored terminal output
    for file logging where escape codes would clutter the output.

    Parameters
    ----------
    Str : str
        Input string potentially containing ANSI escape codes

    Returns
    -------
    str
        String with all ANSI color codes removed

    Examples
    --------
    >>> from ml.pes_trf import ANSI
    >>> colored_str = ANSI.BLUE + "Hello World" + ANSI.RESET
    >>> plain_str = strip_colour(colored_str)
    >>> # plain_str is now "Hello World" without escape sequences

    Notes
    -----
    • Inspects global ANSI object for available color codes
    • Removes all attributes from ANSI that don't start with underscore
    • Safe to call on already-plain strings (returns unchanged)
    • Used internally by tee() to prepare file output
    """
    AnsiColours = [i for i in dir(ANSI) if not i.startswith('_')]
    for Colour in AnsiColours:
        Str = Str.replace(getattr(ANSI, Colour), '')
    return Str
