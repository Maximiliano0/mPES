"""
Logging Utilities for pes_a2c (Pandemic Experiment Scenario — Advantage Actor-Critic)

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
        Format: PES_log_{SubjectId}.txt

    Returns
    -------
    file
        File handle opened in write mode with UTF-8 encoding

    Raises
    ------
    AssertionError
        If SubjectId is None

    Examples
    --------
    >>> handle = create_ConsoleLog_filehandle_singleton('2026-02-09_A2C_AGENT')
    >>> # Creates: outputs/PES_log_2026-02-09_A2C_AGENT.txt
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
    """
    global ConsoleLog_filehandle
    assert ConsoleLog_filehandle is not None, "ConsoleLog_filehandle provided was None. Aborting `close` operation."
    ConsoleLog_filehandle.close()


def _convert_tensorflow_types(value):
    """
    Convert TensorFlow tensor objects to native Python types.

    Parameters
    ----------
    value : any
        Value potentially containing TensorFlow objects

    Returns
    -------
    any
        Converted value (native Python type) or original value if not TensorFlow
    """
    if hasattr(value, 'numpy'):  # TensorFlow Variable or Tensor
        return value.numpy().item() if value.numpy().ndim == 0 else value.numpy()
    return value


def tee(*Strings, **kwargs):
    """
    Dual-stream logging: print to terminal AND write timestamped output to log file.

    Simultaneously outputs a message to both the console (with ANSI colors) and
    the log file (with UTC timestamp, colors stripped).

    Parameters
    ----------
    *Strings : str or convertible
        Variable number of values to log.
    **kwargs : dict, optional
        Keyword arguments passed to print()

    Raises
    ------
    AssertionError
        If ConsoleLog_filehandle has not been initialized
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

    Parameters
    ----------
    Str : str
        Input string potentially containing ANSI escape codes

    Returns
    -------
    str
        String with all ANSI color codes removed
    """
    AnsiColours = [i for i in dir(ANSI) if not i.startswith('_')]
    for Colour in AnsiColours:
        Str = Str.replace(getattr(ANSI, Colour), '')
    return Str
