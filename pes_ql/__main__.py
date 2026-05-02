"""
Main entry point for the pes_ql experiment.

Defines the main() function that orchestrates the full experiment lifecycle:
validation of RL-Agent training files, session creation with logging,
block/sequence/trial assignment, RL-Agent decision collection via
pygameMediator, severity updates, performance calculation, and
result report generation (JSON + PNG).

This module is structurally identical to pes/__main__.py.  The Bayesian-specific
behaviour (hyperparameter search) is performed offline via ext/optimize_rl.py;
the resulting Q-table is consumed here at experiment time.

Experiment Structure
--------------------
::

  Experimento (1)
  ├─ Bloque (8)
  │  ├─ Secuencia / Mapa (8)
  │  │  ├─ Trial / Ciudad (3~10)
  │  │  │  └─ Decisión de Recursos (0-10)

Summary
-------
- 1 Experimento
- 8 Bloques (NUM_BLOCKS)
- 8 Secuencias por Bloque (NUM_SEQUENCES)
- 3-10 Trials por Secuencia (NUM_MIN_TRIALS - NUM_MAX_TRIALS)
- ~360 trials totales, ~45 por bloque (TOTAL_NUM_TRIALS_IN_BLOCK)

Usage
-----
::

    python3 -m pes_ql

Configuration
-------------
All experiment parameters come from config/CONFIG.py.
"""

##############################################################
## Ensure we're running in package mode before proceeding!  ##
##############################################################

assert __package__ is not None and len(__package__) > 0, """
The '__main__' module does not seem to have been run in the context of a runnable package.
Did you forget to add the '-m' flag?

Usage: python3 -m <packagename>
"""

######################
## External Imports ##
######################
import os
import numpy
import datetime

# Force TensorFlow to use CPU by default before any TF import happens.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

######################
## Internal Imports ##
######################
from . import (AGGREGATION_METHOD, ANSI, AVAILABLE_RESOURCES_PER_SEQUENCE,
               INITIAL_SEVERITY_FILE, INIT_NO_OF_CITIES, INPUTS_PATH,
               MAX_ALLOCATABLE_RESOURCES, MAX_INIT_RESOURCES, MAX_INIT_SEVERITY,
               MIN_ALLOCATABLE_RESOURCES, MIN_INIT_RESOURCES, MIN_INIT_SEVERITY,
               NUM_ATTEMPTS_TO_ASSIGN_SEQ, NUM_BLOCKS, NUM_MAX_TRIALS, NUM_MIN_TRIALS,
               NUM_SEQUENCES, OUTPUT_FILE_PREFIX, OUTPUTS_PATH, PLAYER_TYPE,
               RANDOM_INITIAL_SEVERITY, SAVE_INITIAL_SEVERITY_TO_FILE, SAVE_RESULTS,
               STARTING_BLOCK_INDEX, STARTING_SEQ_INDEX, TOTAL_NUM_TRIALS_IN_BLOCK,
               TRUST_MAX, USE_FIXED_BLOCK_SEQUENCES)
from . src import exp_utils
from . src import log_utils
from . src import pygameMediator
from . src import result_formatter
from . src import terminal_utils

#################################################
##  Initialization of Module-global variables  ##
#################################################
Responses_filehandle = None
Starting_sequence_cumulative_index = STARTING_BLOCK_INDEX * NUM_SEQUENCES + STARTING_SEQ_INDEX

call_nominated_aggregator = {
    'confidence_weighted_median': exp_utils.get_confidence_weighted_median,
    'confidence_weighted_mean':   exp_utils.get_confidence_weighted_mean,
    'confidence_weighted_mode':   exp_utils.get_confidence_weighted_mode,
}[AGGREGATION_METHOD]

###################################
##             Main             ###
###################################


def main():
    """Orchestrate the full pes_ql experiment lifecycle.

    Validates the trained Q-Learning artefacts (``q.npy`` and
    ``rewards.npy``) under ``INPUTS_PATH``, creates a dated session
    directory under ``OUTPUTS_PATH`` with a tee'd console log, assigns
    block/sequence/trial structure (8 blocks × 8 sequences × 3–10
    trials), drives the RL agent through every trial via
    ``pygameMediator.provide_rl_agent_response``, computes per-sequence
    normalised performance, generates the JSON + PNG result reports,
    and prints a one-line ``raw_mean_perf`` consistency check against
    ``best_params.json['mean_perf']``.

    Returns
    -------
    None
        All artefacts are written to disk; nothing is returned.
    """

    global Responses_filehandle

    # Validate RL-Agent training files if needed
    if PLAYER_TYPE == 'RL_AGENT':
        q_file = os.path.join(INPUTS_PATH, 'q.npy')
        rewards_file = os.path.join(INPUTS_PATH, 'rewards.npy')

        terminal_utils.header("RL-AGENT VALIDATION", width=80)
        terminal_utils.info("Checking training files...")

        if not os.path.isfile(q_file):
            terminal_utils.error("Q-Table file not found!")
            terminal_utils.list_item(f"Expected path: {q_file}", level=2)
            print("\nTo train the RL-Agent, run:")
            terminal_utils.list_item("python3 -m pes_ql.ext.train_rl")
            print()
            return

        if not os.path.isfile(rewards_file):
            terminal_utils.error("Rewards history file not found!")
            terminal_utils.list_item(f"Expected path: {rewards_file}", level=2)
            print("\nTo train the RL-Agent, run:")
            terminal_utils.list_item("python3 -m pes_ql.ext.train_rl")
            print()
            return

        # Validate that files can be loaded
        try:
            Q = numpy.load(q_file)
            rewards = numpy.load(rewards_file)
            terminal_utils.success("Q-Table loaded successfully")
            terminal_utils.list_item(f"Shape: {Q.shape}", level=2)
            terminal_utils.list_item(f"Data type: {Q.dtype}", level=2)

            terminal_utils.success("Rewards history loaded successfully")
            terminal_utils.list_item(f"Episodes: {len(rewards)}", level=2)
            terminal_utils.list_item(f"Data type: {rewards.dtype}", level=2)
            print()
        except Exception as e:
            terminal_utils.error("Failed to load training files!")
            terminal_utils.list_item(f"Error: {str(e)}", level=2)
            print("\nPlease retrain the model by running:")
            terminal_utils.list_item("python3 -m pes_ql.ext.train_rl")
            print()
            return

        terminal_utils.success("All training files validated successfully!")
        print()

    # Defaults for variables populated inside the SAVE_RESULTS block
    MyPerformances: list = []
    MySubjectId = ""
    session_outputs_path = ""
    total_movement: dict = {}
    Responses_filehandle = None

    # --------------------------------- #
    #       Files to be written         #
    # --------------------------------- #
    if SAVE_RESULTS:

        # MySubjectId: a unique identifier for the current experiment session
        experiment_date = datetime.date.today().strftime("%Y-%m-%d")
        MySubjectId = f"{experiment_date}_{PLAYER_TYPE}"

        # Create experiment-specific folder
        session_outputs_path = os.path.join(OUTPUTS_PATH, MySubjectId)
        os.makedirs(session_outputs_path, exist_ok=True)
        log_utils.create_ConsoleLog_filehandle_singleton(MySubjectId)

        terminal_utils.header(f"EXPERIMENT: {PLAYER_TYPE.upper()}", width=80)
        terminal_utils.info(f"Session ID: {MySubjectId}")
        terminal_utils.info(f"Output directory: {session_outputs_path}")
        print()

        # Save experiment configuration parameters to file
        SubjectInfo_filename = os.path.join(
            session_outputs_path,
            f'{OUTPUT_FILE_PREFIX}{MySubjectId}.txt')

        SubjectInfo_filehandle = open(SubjectInfo_filename, 'w')

        SubjectInfo_filehandle.write("=" * 80 + "\n")
        SubjectInfo_filehandle.write("EXPERIMENT CONFIGURATION PARAMETERS\n")
        SubjectInfo_filehandle.write("=" * 80 + "\n\n")
        SubjectInfo_filehandle.write(f"{'Variable Name':<45} {'Value':<35}\n")
        SubjectInfo_filehandle.write("-" * 80 + "\n")

        config_params = [
            ('AVAILABLE_RESOURCES_PER_SEQUENCE', AVAILABLE_RESOURCES_PER_SEQUENCE),
            ('INIT_NO_OF_CITIES', INIT_NO_OF_CITIES),
            ('INITIAL_SEVERITY_FILE', INITIAL_SEVERITY_FILE),
            ('MAX_ALLOCATABLE_RESOURCES', MAX_ALLOCATABLE_RESOURCES),
            ('MAX_INIT_RESOURCES', MAX_INIT_RESOURCES),
            ('MAX_INIT_SEVERITY', MAX_INIT_SEVERITY),
            ('MIN_ALLOCATABLE_RESOURCES', MIN_ALLOCATABLE_RESOURCES),
            ('MIN_INIT_RESOURCES', MIN_INIT_RESOURCES),
            ('MIN_INIT_SEVERITY', MIN_INIT_SEVERITY),
            ('NUM_ATTEMPTS_TO_ASSIGN_SEQ', NUM_ATTEMPTS_TO_ASSIGN_SEQ),
            ('NUM_BLOCKS', NUM_BLOCKS),
            ('NUM_MIN_TRIALS', NUM_MIN_TRIALS),
            ('NUM_MAX_TRIALS', NUM_MAX_TRIALS),
            ('NUM_SEQUENCES', NUM_SEQUENCES),
            ('PLAYER_TYPE', PLAYER_TYPE),
            ('STARTING_BLOCK_INDEX', STARTING_BLOCK_INDEX),
            ('STARTING_SEQ_INDEX', STARTING_SEQ_INDEX),
            ('TOTAL_NUM_TRIALS_IN_BLOCK', TOTAL_NUM_TRIALS_IN_BLOCK),
            ('TRUST_MAX', TRUST_MAX),
            ('USE_FIXED_BLOCK_SEQUENCES', USE_FIXED_BLOCK_SEQUENCES),
        ]

        for param_name, param_value in config_params:
            SubjectInfo_filehandle.write(f"{param_name:<45} {param_value}\n")

        SubjectInfo_filehandle.write("\n" + "=" * 80 + "\n")

        SubjectInfo_filehandle.close()

        # Save responses to file
        Responses_filename = os.path.join(
            session_outputs_path,
            f'{OUTPUT_FILE_PREFIX}responses_{MySubjectId}.txt'
        )

        Responses_filehandle = open(Responses_filename, 'w')

        # ---------------------------------#
        # MapIndex and sequences assigment #
        # ---------------------------------#
        response = {}
        hold_response_times = {}
        release_response_times = {}
        confidence = {}
        total_movement = {}

        NumTrials__blocks_x_sequences__2darray = numpy.zeros((NUM_BLOCKS, NUM_SEQUENCES))   # each slot contains the
        # number of trials (i.e.
        # cities) that will be
        # presented for a particular
        # 'block,sequence' pair. (Note
        # that these numbers do not
        # include the 'initial cities
        # on map'; only cities for
        # which the player will have
        # to make an allocation
        # decision)

        MapIndices__blocks_x_sequences__2darray = numpy.zeros((NUM_BLOCKS, NUM_SEQUENCES))  # each slot contains a
        # map index (from 0 to 8
        # which background image
        # file will be selected
        # for a particular
        # 'block,sequence' pair

        NumTrialsPerSequence_list = []   # this list contains exactly the same information as the
        # NumTrials__blocks_x_sequences__2darray array, but as a flat list of integers,
        # rather than as an nblocks-by-nsequences array.

        # NOTE: The following assignment could potentially be erased (?)
        pygameMediator.number_of_trials = NumTrials__blocks_x_sequences__2darray

        # Block (blk) into the Experiment (MySubjectId)
        for blk in range(NUM_BLOCKS):

            numpy.random.seed(100 + blk)

            # Sequence (seq) into the Block (blk)
            for seq in range(NUM_SEQUENCES):

                counter_seq = NUM_ATTEMPTS_TO_ASSIGN_SEQ

                # Assign a random map index to this block-sequence pair,
                # ensuring it is not repeated within the same block
                while counter_seq > 0:

                    b = numpy.random.randint(0, 9)

                    if b not in MapIndices__blocks_x_sequences__2darray[blk, :]:
                        MapIndices__blocks_x_sequences__2darray[blk, seq] = b
                        break

                    counter_seq -= 1

                if (USE_FIXED_BLOCK_SEQUENCES):
                    NumTrials__blocks_x_sequences__2darray[blk, :] = exp_utils.next_seq_length(
                        blk * NUM_SEQUENCES,
                        NUM_SEQUENCES
                    )
                else:
                    NumTrials__blocks_x_sequences__2darray[blk, :] = exp_utils.sampler(
                        NUM_SEQUENCES,
                        TOTAL_NUM_TRIALS_IN_BLOCK,
                        [NUM_MIN_TRIALS, NUM_MAX_TRIALS],
                        rn=blk
                    )

                _counter_trial = INIT_NO_OF_CITIES + NumTrials__blocks_x_sequences__2darray[blk, seq]

                NumTrialsPerSequence_list.append(NumTrials__blocks_x_sequences__2darray[blk, seq])

            # END OF for seq in range( NUM_SEQUENCES ):

        # END OF `for blk in range( NUM_BLOCKS )`

        # ----------------------------- #
        # Initial severity assignation  #
        # ----------------------------- #
        InitialSeverityCsv = os.path.join(INPUTS_PATH, INITIAL_SEVERITY_FILE)

        if RANDOM_INITIAL_SEVERITY:
            first_severity = exp_utils.random_severity_generator(
                int(numpy.sum(NumTrials__blocks_x_sequences__2darray)), 2, 9)
            if SAVE_INITIAL_SEVERITY_TO_FILE:
                numpy.savetxt(InitialSeverityCsv, first_severity, fmt='%d', delimiter=',')
        else:
            first_severity = numpy.loadtxt(InitialSeverityCsv)
            first_severity = first_severity[0: int(numpy.sum(NumTrials__blocks_x_sequences__2darray))]

        # NOTE: The following assignment could potentially be erased (?)
        pygameMediator.first_severity = first_severity

        # Initialise output (i.e. responses file) by creating a header
        Responses_filehandle.write("#InitialSeverity, Response, Confidence, PressEvent_seconds, ReleaseEvent_seconds\n")
        Responses_filehandle.flush()

        # ----------------------- #
        # Experimental procedure  #
        # ----------------------- #
        print("--- Starting experiment --- \n")
        total_number_of_sequences = NUM_BLOCKS * NUM_SEQUENCES

        AbsoluteTrialCount = 0
        AbsoluteSequenceIndex = 0
        ExperimentStartTime = datetime.datetime.now(tz=datetime.timezone.utc)

        MyPerformances = []
        # A blockindex-by-playerindex array (only other players). Initial value: TRUST_MAX (i.e. full trust)
        TrustRatings = numpy.array([1, TRUST_MAX])

        # Print experiment start banner
        log_utils.tee(
            f"\n{'=' * 100}"
        )
        log_utils.tee(
            f"{ANSI.BOLD}{ANSI.GREEN}  STARTING EXPERIMENT: {PLAYER_TYPE}{ANSI.RESET}"
        )
        log_utils.tee(
            f"  Total: {NUM_BLOCKS} Blocks × {NUM_SEQUENCES} Sequences = {NUM_BLOCKS * NUM_SEQUENCES} Sessions"
        )
        log_utils.tee(
            f"{'=' * 100}\n"
        )

        # Block (CurrentBlockIndex) into the Experiment (MySubjectId)
        for CurrentBlockIndex in range(NUM_BLOCKS):

            # Jump to the block and sequence specified in the config, if present
            if CurrentBlockIndex < STARTING_BLOCK_INDEX:
                AbsoluteTrialCount += int(NumTrials__blocks_x_sequences__2darray[CurrentBlockIndex, :].sum())
                AbsoluteSequenceIndex += NUM_SEQUENCES
                continue
            else:
                pass  # We have (already) reached the correct starting block

            log_utils.tee(
                f"\n{ANSI.BOLD}{ANSI.BLUE}▶ BLOQUE {CurrentBlockIndex + 1} / {NUM_BLOCKS}{ANSI.RESET}"
            )

            # Obtain current time in experiment
            TimeElapsed = (datetime.datetime.now(tz=datetime.timezone.utc) - ExperimentStartTime).total_seconds()
            HoursElapsed = max(0, TimeElapsed / 60 / 60 // 1)
            MinutesElapsed = max(0, (TimeElapsed - HoursElapsed * 60 * 60) / 60 // 1)
            HoursElapsed_str = (
                f'{HoursElapsed:.0f} hours, and ' if HoursElapsed > 1
                else '1 hour, and ' if HoursElapsed == 1 else ''
            )
            MinutesElapsed_str = (
                f'{MinutesElapsed:.0f} minutes' if MinutesElapsed != 1
                else '1 minute' if MinutesElapsed == 1 else ''
            )
            TimeElapsed_str = f'Time elapsed: {HoursElapsed_str}{MinutesElapsed_str}'

            # Calculate time left based on speed so far and sequences remaining
            if AbsoluteSequenceIndex > Starting_sequence_cumulative_index:
                TimePerSeq = TimeElapsed / (AbsoluteSequenceIndex - Starting_sequence_cumulative_index)
                SeqsLeft = total_number_of_sequences - AbsoluteSequenceIndex
                TimeLeft = SeqsLeft * TimePerSeq
                HoursLeft = max(0, TimeLeft / 60 / 60 // 1)
                MinutesLeft = max(0, (TimeLeft - HoursLeft * 60 * 60) / 60 // 1)
                HoursLeft_str = (
                    f'{HoursLeft:.0f} hours, and ' if HoursLeft > 1
                    else '1 hour, and ' if HoursLeft == 1 else ''
                )
                MinutesLeft_str = (
                    f'{MinutesLeft:.0f} minutes' if MinutesLeft != 1
                    else '1 minute' if MinutesLeft == 1 else ''
                )
                TimeLeft_str = f'Estimated time left: {HoursLeft_str}{MinutesLeft_str}'
                _EndOfBlock_str = (
                    f'End of block {CurrentBlockIndex}   '
                    f'({NUM_BLOCKS - CurrentBlockIndex} blocks to go)\nTake rest if needed.'
                )
            else:
                _EndOfBlock_str = ' '
                TimeLeft_str = ' '

            log_utils.tee(f'{TimeElapsed_str} -- {TimeLeft_str}')

            # Get map indices for all sequences in this block
            CurrentBlockMapIndices = MapIndices__blocks_x_sequences__2darray[CurrentBlockIndex, :]

            response[CurrentBlockIndex] = {}
            hold_response_times[CurrentBlockIndex] = {}
            release_response_times[CurrentBlockIndex] = {}
            confidence[CurrentBlockIndex] = {}
            total_movement[CurrentBlockIndex] = {}

            # Start interactive processing of sequences
            for CurrentSequenceIndex, CurrentSequenceMapIndex in enumerate(CurrentBlockMapIndices):

                if CurrentBlockIndex == STARTING_BLOCK_INDEX and CurrentSequenceIndex < STARTING_SEQ_INDEX:
                    AbsoluteTrialCount += int(
                        NumTrials__blocks_x_sequences__2darray[CurrentBlockIndex, CurrentSequenceIndex])
                    AbsoluteSequenceIndex += 1
                    continue

                resources_to_allocate = AVAILABLE_RESOURCES_PER_SEQUENCE

                log_utils.tee(
                    f"  └─ Secuencia {AbsoluteSequenceIndex + 1}/{total_number_of_sequences}"
                    f" (Mapa #{int(CurrentSequenceMapIndex)})"
                    f" - Progreso: {CurrentSequenceIndex + 1}/{NUM_SEQUENCES}"
                )

                _ScreenMessage = (f"Map Number #{AbsoluteSequenceIndex + 1} / {total_number_of_sequences}\n\n\n"
                                 f"Press the LEFT MOUSE BUTTON to start.\n\n\n"
                                 f"(Note: If you need to terminate the\n"
                                 f"experiment prematurely, press ESC now.)"
                                 )

                # Initialise arrays that will hold the participant's various outputs of interest
                response[CurrentBlockIndex][CurrentSequenceIndex] = []
                hold_response_times[CurrentBlockIndex][CurrentSequenceIndex] = []
                release_response_times[CurrentBlockIndex][CurrentSequenceIndex] = []
                confidence[CurrentBlockIndex][CurrentSequenceIndex] = []
                total_movement[CurrentBlockIndex][CurrentSequenceIndex] = []

                # Initialise arrays that will hold map-specific information for this sequence
                init_severity = []   # Initial severity allocated to each city in the map BEFORE EACH TRIAL
                ResourceAllocationsAtCurrentlyVisibleCities = []   # Resources currently allocated to each city in the map

                # Initialise the map with the two pre-trial cities and their severities
                # (following random resource allocation)
                numpy.random.seed(3)  # NOTE: In practice, this seed, given an INIT_NO_OF_CITIES of 2, always results in
                # initial city severities of 4 and 3, and initial resource allocations to those
                # cities of 3 and 6. Meaning the player always has 40 resources left to allocate to
                # the remainder of the sequence.

                for c in range(INIT_NO_OF_CITIES):

                    init_severity.append(numpy.random.randint(MIN_INIT_SEVERITY, 1 + MAX_INIT_SEVERITY))
                    ResourceAllocationsAtCurrentlyVisibleCities.append(
                        numpy.random.randint(MIN_INIT_RESOURCES, 1 + MAX_INIT_RESOURCES))

                resources_left = resources_to_allocate - numpy.sum(ResourceAllocationsAtCurrentlyVisibleCities)
                # NOTE: resources_left: i.e. in this sequence (i.e. as opposed to in block)
                # NOTE: resources_to_allocate == AVAILABLE_RESOURCES_PER_SEQUENCE == 39

                SeveritiesOfCurrentlyVisibleCities = exp_utils.get_updated_severity(INIT_NO_OF_CITIES,
                                                                                    ResourceAllocationsAtCurrentlyVisibleCities,
                                                                                    init_severity)

                # Update whether severity has increased or decreased per city
                direction = []
                for i in range(len(SeveritiesOfCurrentlyVisibleCities)):
                    if SeveritiesOfCurrentlyVisibleCities[i] < init_severity[i]:
                        direction.append(2)   # decrease in severity
                    else:
                        direction.append(1)   # increase in severity

                # Show End Of Trial Feedback -- will be turned off 'after' resources end.
                _ShowEndOfTrialFeedback = True

                # Pre-initialise message array (overwritten each trial)
                MyMessage = numpy.zeros((0, 3))

                # Map initialisation complete; now proceed with Human/AI Agent trial annotations
                for trial_no in range(
                        int(NumTrials__blocks_x_sequences__2darray[CurrentBlockIndex, CurrentSequenceIndex])):

                    AbsoluteTrialCount += 1
                    AbsoluteTrialIndex = AbsoluteTrialCount - 1

                    # i.e. severity before update (used in the update step)
                    init_severity = SeveritiesOfCurrentlyVisibleCities

                    log_utils.tee()   # will print an empty line

                    log_utils.tee(
                        f'Current trial: {trial_no + 1} out of'
                        f' {NumTrials__blocks_x_sequences__2darray[CurrentBlockIndex, CurrentSequenceIndex]:.0f} in sequence',
                        f'({AbsoluteTrialCount} out of {TOTAL_NUM_TRIALS_IN_BLOCK * NUM_BLOCKS} overall)'
                    )

                    log_utils.tee(
                        'Initial severity values before annotation: ',
                        ", ".join([f'{i:.2f}' for i in init_severity])
                    )

                    log_utils.tee(
                        'Resources remaining: ', int(resources_left) if hasattr(
                            resources_left, 'numpy') else resources_left
                    )

                    new_locations = INIT_NO_OF_CITIES + trial_no + 1   # Number of cities to appear on map in this step
                    # I.e. from random pre-initialization
                    severity_new_location = first_severity[AbsoluteTrialIndex]

                    log_utils.tee(f"Current trial's initial severity: {severity_new_location}")

                    # Severity values of cities on map, before update (used
                    init_severity.append(severity_new_location)
                    # in the update step).

                    ResourceAllocationsAtCurrentlyVisibleCities.append(-1)   # NOTE: purpose of this sentinel append is unclear

                    # Get response from Agent
                    if resources_left > 0 and PLAYER_TYPE == 'RL_AGENT':
                        (pc,      # corresponds to the value of 'confidence' precalculated on provide_response.  Ignored for online players.
                         r,
                         # corresponds to the value of 'resp' variable (defined as global within
                         # 'provide_response') at the time of return
                         rt_h,
                         # corresponds to the value of 'rt_hold'       (defined as global within
                         # 'provide_response') at the time of return
                         rt_rel,
                         # corresponds to the value of 'rt_release'    (defined as global within
                         # 'provide_response') at the time of return
                         mov      # corresponds to 'movement' array in 'provide_response'
                         ) = pygameMediator.provide_rl_agent_response(
                            ResourceAllocationsAtCurrentlyVisibleCities,
                            resources_left,
                            CurrentBlockIndex,
                            CurrentSequenceIndex,
                            trial_no
                        )

                        # Get confidence rating from user
                        if r == 0.0:
                            c = -1      # c here stands for 'confidence', not 'city' as elsewhere.
                            mov = [0]
                        else:
                            c = pc

                        confidence[CurrentBlockIndex][CurrentSequenceIndex].append(c)

                        response[CurrentBlockIndex][CurrentSequenceIndex].append(r)
                        hold_response_times[CurrentBlockIndex][CurrentSequenceIndex].append(rt_h)
                        release_response_times[CurrentBlockIndex][CurrentSequenceIndex].append(rt_rel)
                        total_movement[CurrentBlockIndex][CurrentSequenceIndex].append(mov)

                        log_utils.tee(f'Response was: {r}')
                        log_utils.tee(f'Confidence was: {c}')
                        log_utils.tee(f'PressEvent_seconds was: {rt_h}')
                        log_utils.tee(f'ReleaseEvent_seconds was: {rt_rel}')

                    else:   # i.e. resources_left <= 0 or not RL-Agent
                        confidence[CurrentBlockIndex][CurrentSequenceIndex].append(-1)
                        response[CurrentBlockIndex][CurrentSequenceIndex].append(0)
                        hold_response_times[CurrentBlockIndex][CurrentSequenceIndex].append(0)
                        release_response_times[CurrentBlockIndex][CurrentSequenceIndex].append(0)
                        total_movement[CurrentBlockIndex][CurrentSequenceIndex].append([0])

                    if SAVE_RESULTS:

                        Responses_filehandle.write(
                            ", ".join([
                                f"{first_severity[AbsoluteTrialIndex]: >{len('#InitialSeverity')}}",
                                f"{response[CurrentBlockIndex][CurrentSequenceIndex][-1]: >{len('Response')}}",
                                f"{confidence[CurrentBlockIndex][CurrentSequenceIndex][-1]: >{len('Confidence')}}",
                                f"{hold_response_times[CurrentBlockIndex][CurrentSequenceIndex][-1]: >{len('PressEvent_seconds')}}",
                                f"{release_response_times[CurrentBlockIndex][CurrentSequenceIndex][-1]: >{len('ReleaseEvent_seconds')}}"
                            ]) + "\n"
                        )

                        Responses_filehandle.flush()

                    # Send a message to all other players (contains response, confidence, and final severities).
                    _NumTrialsInSequence = int(NumTrialsPerSequence_list[AbsoluteSequenceIndex])
                    StartingIndex = int(sum(NumTrialsPerSequence_list[: AbsoluteSequenceIndex]))
                    InitialSeveritiesInSequence = first_severity[StartingIndex: AbsoluteTrialCount].copy()

                    MyMessage = numpy.c_[
                        response[CurrentBlockIndex][CurrentSequenceIndex],
                        confidence[CurrentBlockIndex][CurrentSequenceIndex],
                        exp_utils.get_array_of_sequence_severities_from_allocations(
                            response[CurrentBlockIndex][CurrentSequenceIndex],
                            InitialSeveritiesInSequence,
                        ),
                    ]

                    # Past cities severity is
                    if resources_left > 0:
                        # Calculate severity and update necessary structures
                        ResourceAllocationsAtCurrentlyVisibleCities[-1] = int(MyMessage[-1, 0])
                        SeveritiesOfCurrentlyVisibleCities = exp_utils.get_updated_severity(
                            new_locations, ResourceAllocationsAtCurrentlyVisibleCities, init_severity)

                        if SeveritiesOfCurrentlyVisibleCities[-1] < init_severity[-1]:
                            direction.append(2)   # decrease in severity
                        else:
                            direction.append(1)   # increase in severity

                        resources_left = int(resources_left - MyMessage[-1, 0])

                    else:
                        ResourceAllocationsAtCurrentlyVisibleCities[-1] = 0
                        SeveritiesOfCurrentlyVisibleCities = exp_utils.get_updated_severity(
                            new_locations, ResourceAllocationsAtCurrentlyVisibleCities, init_severity)

                        if SeveritiesOfCurrentlyVisibleCities[-1] < init_severity[-1]:
                            direction.append(2)   # decrease in severity
                        else:
                            direction.append(1)   # increase in severity

                # END OF `for trial_no in range( int(
                # NumTrials__blocks_x_sequences__2darray[ CurrentBlockIndex,
                # CurrentSequenceIndex ] ) )`

                _NumTrialsInSequence = int(NumTrialsPerSequence_list[AbsoluteSequenceIndex])
                StartingIndex = int(sum(NumTrialsPerSequence_list[: AbsoluteSequenceIndex]))
                InitialSeveritiesInSequence = first_severity[StartingIndex: AbsoluteTrialCount].copy()
                (MyPerformance,
                 _WorstCaseSequenceSeverity,
                 _BestCaseSequenceSeverity) = exp_utils.calculate_normalised_final_severity_performance_metric(
                    MyMessage[:, 2],
                    InitialSeveritiesInSequence
                )
                MyPerformances.append(MyPerformance)
                log_utils.tee(
                    "My Sequence Performances: ",
                    ", ".join([f'{i:.2f}' for i in MyPerformances])
                )
                log_utils.tee()

                # Add the aggregated performance.
                AllMessages = [MyMessage]
                _aggregated_allocations, aggregated_final_severity = call_nominated_aggregator(
                    AllMessages, first_severity, AbsoluteSequenceIndex, AbsoluteTrialCount)
                AggregatedPerformance, *_ = exp_utils.calculate_normalised_final_severity_performance_metric(
                    aggregated_final_severity,
                    InitialSeveritiesInSequence
                )

                # Plot all the accumulated performance for all the players.
                _StartingAbsoluteSequence = STARTING_BLOCK_INDEX * NUM_SEQUENCES + STARTING_SEQ_INDEX

                # For RL-Agent, print results to console
                log_utils.tee(f"Sequence {AbsoluteSequenceIndex}: Performance = {AggregatedPerformance:.4f}")

                AbsoluteSequenceIndex += 1

            # END OF `for CurrentSequenceIndex, CurrentSequenceMapIndex in enumerate( CurrentBlockMapIndices )`

            # Update TrustRatings (skip for RL-Agent)
            log_utils.tee("Trust Ratings:\n", TrustRatings)

        #
        # END OF `for CurrentBlockIndex in range( NUM_BLOCKS )`

    # Generate result summary JSON and visualization PNG
    if PLAYER_TYPE == 'RL_AGENT' and len(MyPerformances) > 0:
        try:
            # Reorganize performances by block for better statistical analysis
            performances_by_block = []
            perf_idx = 0
            for _block_idx in range(NUM_BLOCKS):
                block_performances = []
                for _seq_idx in range(NUM_SEQUENCES):
                    if perf_idx < len(MyPerformances):
                        block_performances.append(MyPerformances[perf_idx])
                        perf_idx += 1
                if block_performances:
                    performances_by_block.append(block_performances)

            # Prepare resource allocation data for comparison
            resource_data = {
                'total_resources_per_sequence': AVAILABLE_RESOURCES_PER_SEQUENCE,
                'agent_type': PLAYER_TYPE,
                'num_blocks': NUM_BLOCKS,
                'num_sequences': NUM_SEQUENCES,
                'total_trials': len(MyPerformances)
            }

            json_path, png_path = result_formatter.generate_results_report(
                MySubjectId,
                session_outputs_path,
                MyPerformances,
                performances_by_block,
                resource_data
            )
            log_utils.tee(f"\n✓ Results summary JSON: {json_path}")
            log_utils.tee(f"✓ Results visualization PNG: {png_path}\n")

            # Print the raw 64-sequence mean explicitly so it can be compared
            # one-glance against ``best_params.json['mean_perf']`` (the metric
            # Optuna reports).
            try:
                _raw_mean = float(numpy.mean(MyPerformances))
                _raw_std = float(numpy.std(MyPerformances))
                log_utils.tee(
                    f"raw_mean_perf = {_raw_mean:.6f}  "
                    f"(std={_raw_std:.6f}, n={len(MyPerformances)})"
                )
                _bp_path = os.path.join(INPUTS_PATH, 'best_params.json')
                if os.path.isfile(_bp_path):
                    import json as _json
                    with open(_bp_path, 'r', encoding='utf-8') as _f:
                        _bp = _json.load(_f)
                    _expected = float(_bp.get('mean_perf', float('nan')))
                    if not numpy.isnan(_expected):
                        _delta = abs(_raw_mean - _expected)
                        log_utils.tee(
                            f"best_params.json mean_perf = {_expected:.6f}  "
                            f"|Δ| = {_delta:.6f}"
                        )
            except Exception as _cmp_err:  # pylint: disable=broad-except
                log_utils.tee(
                    f"Warning: could not compute raw_mean_perf comparison: {_cmp_err}"
                )
        except Exception as e:
            log_utils.tee(f"Warning: Could not generate result reports: {str(e)}")

    # Insert 'successful completion' marker in logfile.
    if SAVE_RESULTS:
        exp_utils.exit_experiment_gracefully(
            Message="",
            Filehandles=[Responses_filehandle],
            MovementData=(
                os.path.join(
                    session_outputs_path,
                    f'{OUTPUT_FILE_PREFIX}movement_log_{MySubjectId}.npy'),
                total_movement),
            LogUtils=log_utils,
            _PygameMediator=pygameMediator
        )

    return
#
# END OF 'main()


if __name__ == '__main__':
    main()
