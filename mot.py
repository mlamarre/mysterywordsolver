"""Script to solve the mystery word game"""
import numpy as np
import argparse
import logging
import cProfile
import pstats
import datetime
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool

ENABLE_MP = True

def numpy_s1_to_str(array):
    return array.tobytes().decode('ascii')

def numpy_s1_2d_array_to_str_list(array):
    return [b''.join(row).decode('ascii') for row in array]

class WordFinderConstraint:
    """Constraint to search in a fixed length word dictionary for a single mystery word.

    The constraint is composed of four sub-constraints:

    1. List of letters NOT found in the word.
    2. List of letters that MUST be in the word.
    3. Position of some of the letters that MUST be in the word.
    4. Unallowed positions for the letters that must be in the word.

    A constraint can be build by a word pair one of which is the mystery word.
    Two constraints can be merged.

    Words are all lower case, fixed size, stored in ASCII encoded bytes in the range [a-z].
    They are numpy arrays for which == and != operators are defined.

    The constraint can be applied in batch on the list of word to filter it down.

    """
    def __init__(self, matched, matched_letters, must_have, must_not_have, forbidden_letters_per_position):
        self.num_letters = matched.shape[0]
        self.matched = matched
        self.matched_letters = matched_letters
        self.must_have = must_have
        self.must_not_have = must_not_have
        self.forbidden_letters_per_position = forbidden_letters_per_position # a num_letters tuple of np.array of letters

    def merge(self, other):
        # if letter was not matched by this but was matched by other, copy the letter from other, otherwise keep the current value
        self.matched_letters = np.where(~self.matched & other.matched, other.matched_letters, self.matched_letters)
        # take the logical OR of the matched letters to combine the constraints
        self.matched = np.logical_or(self.matched, other.matched)
        self.must_not_have = np.union1d(self.must_not_have, other.must_not_have)
        self.must_have = np.union1d(self.must_have, other.must_have)
        for i in range(self.num_letters):
            self.forbidden_letters_per_position[i] = np.union1d(self.forbidden_letters_per_position[i], other.forbidden_letters_per_position[i])
        return self

    def apply(self, words):
        # apply the constraint to a numpy array of words

        # start with the matched letter constraint since it's the probably
        # the fastest in terms of implementation of numpy
        for col, letter in enumerate(self.matched_letters):
            if letter == b'':
                continue
            words = words[words[:, col] == letter]

        # apply the must NOT have constraint 2nd because this set will grow faster
        # and thus reduce the list of words faster
        words = words[np.isin(words, self.must_not_have, assume_unique=True, invert=True).all(axis=1)]

        # apply the must have constraint last
        for i in range(self.must_have.shape[0]):
            words = words[np.isin(words, self.must_have[i], assume_unique=True).any(axis=1)]

        # apply the possible letters per position constraint
        for i in range(self.num_letters):
            if len(self.forbidden_letters_per_position[i]) > 0:
                words = words[np.isin(words[:, i], self.forbidden_letters_per_position[i], assume_unique=True,  invert=True)]

        return words

def constraint_from_word_pair(test_word, mystery_word):
    """Create a constraint from a word pair."""
    matched = test_word == mystery_word
    # put special value 0 for non-matched letters - copy the matching lettes
    matched_letters = np.where(matched, mystery_word, b'')
    found_letters_mask = np.isin(test_word, mystery_word)
    must_have = np.unique(test_word[found_letters_mask])
    must_not_have = np.unique(test_word[~found_letters_mask])
    empty_array = np.array([], dtype='S1')
    num_letters = test_word.shape[0]
    forbidden_letters_per_position = [empty_array for _ in range(num_letters)]
    for i in range(num_letters):
        if not matched[i]:
            forbidden_letters_per_position[i] = np.union1d(forbidden_letters_per_position[i], test_word[i])
    return WordFinderConstraint(matched, matched_letters, must_have, must_not_have, forbidden_letters_per_position)

def constraint_from_command_line_input(test_input, test_output):
    """Create a constraint from a test output.`

    Parameters
    ----------
    test_input : np.ndarray of shape (1, num_letters) of type S1
        A numpy array of bytes representing the test input.
    test_output : str
        A unicode string from the user input.
    """
    # Matched letter are the upper case letters - use python string to find them
    # Get a lower case version of the test output
    test_array = np.frombuffer(test_output.lower().encode('ascii'), dtype='S1')
    matched = np.array([c.isupper() for c in test_output], dtype=bool)
    must_have_mask = np.array([c.islower() for c in test_output], dtype=bool)
    # Letters of test input for which we have a lower case letter must be put in the must_have list combined with the matched letters
    must_have = np.union1d(np.unique(test_input[must_have_mask]), test_input[matched])
    matched_letters = np.where(matched, test_array, b'')
    # Letters of test input for which we have a dot must be put in the must_not_have list
    must_not_have = np.unique(test_input[test_array == b'.'])
    # Special case for too many identical letters in test input verus the mystery word
    # in this case the extra instance of letters for which we got . must be removed from the must_not_have list
    must_not_have = np.setdiff1d(must_not_have, must_have, assume_unique=True)
    # The forbidden letters per position are the input letters for which we have a dot
    empty_array = np.array([], dtype='S1')
    num_letters = test_input.shape[0]
    forbidden_letters_per_position = [empty_array for _ in range(num_letters)]
    for i in range(num_letters):
        if not matched[i]:
            forbidden_letters_per_position[i] = np.union1d(forbidden_letters_per_position[i], test_input[i])
    return WordFinderConstraint(matched, matched_letters, must_have, must_not_have, forbidden_letters_per_position)

def load_dictionary(wordlist_path, num_letters):
    """Load a list of words from a file."""
    if not wordlist_path.exists():
        raise FileNotFoundError(f"File not found: {wordlist_path}")
    with wordlist_path.open("r") as f:
        words = f.readlines()
    logger.info(f"Number of words in the dictionary: {len(words)}")
    valid_words = []
    for word in words:
        word = word.strip().lower()
        if word.isalpha() and len(word) == num_letters:
            valid_words.append(word.encode('ascii'))
    return valid_words

def process_test_word(args):
    i, test_word, word_array = args
    remaining_words_matrix_row = np.zeros(word_array.shape[0], dtype=np.int32)
    for j, mystery_word in enumerate(word_array):
        constraint = constraint_from_word_pair(test_word, mystery_word)
        reduced_word_array = constraint.apply(word_array)
        remaining_words_matrix_row[j] = reduced_word_array.shape[0]
    return i, remaining_words_matrix_row

def find_best_word(test_word_array, remaining_word_array, max_pair_test, logger):
    """Find the best word to use as a test word."""
    # if test_word_array.shape[0] * remaining_word_array.shape[0] > max_pair_test then use random sampling
    if max_pair_test is not None and test_word_array.shape[0] * remaining_word_array.shape[0] > max_pair_test:
        logger.info(f"Random sampling of {max_pair_test} pairs of test words and remaining words.")
        # reduce only the number of test words by random sampling
        ntestwordreduced = int(max_pair_test / remaining_word_array.shape[0])
        test_word_array = test_word_array[np.sort(np.random.choice(test_word_array.shape[0], ntestwordreduced, replace=False))]

    # Do the O(N^2) constraint tests and compute statistics
    remaining_words_matrix = np.zeros((test_word_array.shape[0], remaining_word_array.shape[0]), dtype=np.int32)

    if ENABLE_MP:
        with Pool() as pool:
            results = list(tqdm(pool.imap(process_test_word,
                                        [(i, test_word, remaining_word_array) for i, test_word in enumerate(test_word_array)]),
                                total=len(test_word_array)))

        for i, remaining_words_matrix_row in results:
            remaining_words_matrix[i] = remaining_words_matrix_row
    else:
        for i, test_word in enumerate(tqdm(test_word_array)):
            remaining_words_matrix[i] = process_test_word((i, test_word, remaining_word_array))[1]

    if remaining_words_matrix.shape[1] == 0:
        raise ValueError("No possible word left - invalid test input.")

    # Find the min(max(remaining words)) test word
    min_max_idx = np.argmin(np.max(remaining_words_matrix, axis=1))
    min_max_word = test_word_array[min_max_idx]
    logger.info(f"Min(max(remaining words)) test word: {numpy_s1_to_str(min_max_word)}")
    # Find the min(mean(remaining words)) test word
    min_mean_idx = np.argmin(np.mean(remaining_words_matrix, axis=1))
    min_mean_word = test_word_array[min_mean_idx]
    logger.info(f"Min(mean(remaining words)) test word: {numpy_s1_to_str(min_mean_word)}")

    return min_max_word, min_mean_word

def mystery_word_main(dictionary_file, num_letters, compute_best_words, strategy, max_pair_test, logger):
    """Main function to solve the mystery word game."""

    logger.info(f"Using dictionary file: {dictionary_file}")
    logger.info(f"Number of letters: {num_letters}")

    dictionary_file = Path(dictionary_file)
    words = load_dictionary(dictionary_file, num_letters)
    word_array = np.frombuffer(b''.join(words), dtype='S1').reshape(-1, num_letters)
    logger.info(f"Number of valid words: {word_array.shape[0]}")
    output_folder = dictionary_file.parent

    best_min_max_1st_word_file = output_folder / f"{dictionary_file.stem}_min_max_test_words_{num_letters}.txt"
    best_min_mean_1st_word_file = output_folder / f"{dictionary_file.stem}_min_mean_test_words_{num_letters}.txt"

    # The best first word depends only on the dictionary so it can be computed once and saved
    if strategy == "min_max":
        best_1st_word_file = best_min_max_1st_word_file
    elif strategy == "min_mean":
        best_1st_word_file = best_min_mean_1st_word_file

    if compute_best_words:
        min_max_word, min_mean_word = find_best_word(word_array, word_array, None, logger)
        with best_min_max_1st_word_file.open("w") as f:
            f.write(numpy_s1_to_str(min_max_word) + "\n")
        with best_min_mean_1st_word_file.open("w") as f:
            f.write(numpy_s1_to_str(min_mean_word) + "\n")
        print(f'Best min-max word {numpy_s1_to_str(min_max_word)} and best min-mean word {numpy_s1_to_str(min_mean_word)} saved to file.')
        print("Bye!")
        return

    if best_1st_word_file.exists():
        with best_1st_word_file.open("r") as f:
            test_words = f.readlines()
        test_words = [word.strip().lower().encode('ascii') for word in test_words]
        test_word_array = np.frombuffer(b''.join(test_words), dtype='S1').reshape(-1, num_letters)
        best_1st_word = test_word_array[0]
        print(f"Best ({strategy}) 1st word : {numpy_s1_to_str(best_1st_word)} loaded from file.")

    # Start the game
    mystery_word_array = word_array.copy()

    print('Welcome to the Mystery Word Game Solver!')
    print("The test output should use UPPER CASE for a good position match, lower case for a correct letter but in the wrong position and a dot ('.') for a letter not in the word.")
    print("Example: Test word: 'tear', Mystery word: 'tape', Test output: 'Tea.'")

    # must solve in 6 rounds
    try:
        for i in range(6):
            while True:
                # Ask the user to input its test word on the command line
                test_word = input(f"[{i+1}] Enter your test word ({num_letters} letters): ").strip().lower()
                if len(test_word) != num_letters:
                    logger.error(f"Test word must have {num_letters} letters.")
                    continue
                test_word = np.frombuffer(test_word.encode('ascii'), dtype='S1').reshape(num_letters)
                if not np.isin(test_word, word_array).all():
                    logger.error(f"Invalid word - it must be in the dictionary.")
                    continue
                break

            # Ask the user for the output of the test using an UPPER CASE letter to denote a good position match
            # and a lower case letter to denote a correct letter but in the wrong position and a dot for a letter not in the word
            while True:
                test_output = input(f"[{i+1}] Enter the test output for {numpy_s1_to_str(test_word)}: ").strip()
                if len(test_output) != num_letters:
                    logger.error(f"Test output must have {num_letters} characters.")
                    continue
                break

            constraint = constraint_from_command_line_input(test_word, test_output)
            mystery_word_array = constraint.apply(mystery_word_array)
            print(f"Number of remaining potential words: {mystery_word_array.shape[0]}")

            if mystery_word_array.shape[0] == 1:
                print(f"The mystery word is: {numpy_s1_to_str(mystery_word_array[0])}")
                # save the mystery word to a file to make a list of words to use as a dictionary
                mystery_word_file = output_folder / f"{dictionary_file.stem}_mystery_word_{num_letters}.txt"
                # append to existing text file on last line, save the date and the word with a space in between
                with mystery_word_file.open("a") as f:
                    # write the date
                    f.write(datetime.datetime.now().strftime("%Y-%m-%d") + " ")
                    # write the mystery word
                    f.write(numpy_s1_to_str(mystery_word_array[0]) + "\n")
                break
            elif mystery_word_array.shape[0] == 2:
                print(f"The mystery word is one of two: {numpy_s1_2d_array_to_str_list(mystery_word_array)} - pick one!")
                continue
            elif mystery_word_array.shape[0] <= 10:
                print("Potential words:")
                print(numpy_s1_2d_array_to_str_list(mystery_word_array))

            min_max_word, min_mean_word = find_best_word(word_array, mystery_word_array, max_pair_test, logger)
            if strategy == "min_max":
                best_word = min_max_word
            elif strategy == "min_mean":
                best_word = min_mean_word

            print(f"Best ({strategy}) test word: {numpy_s1_to_str(best_word)}")
    except KeyboardInterrupt:
        print("Bye!")
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a dictionary file.")
    parser.add_argument("dictionary_file", type=str, nargs='?', default="French ODS Dictionary.txt", help="The name of the dictionary file.")
    parser.add_argument("--num_letters", type=int, choices=[4, 5, 6], default=4, help="The number of letters in the words.")
    parser.add_argument("--compute_best_words", action="store_true", help="Compute the best test words.")
    parser.add_argument("--strategy", type=str, choices=["min_max", "min_mean"], default="min_mean", help="The strategy to use to find the test word.")
    parser.add_argument("--log_level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="WARNING", help="Set the logging level.")
    parser.add_argument("--profile", action="store_true", help="Enable profiling.")
    parser.add_argument("--max_pair_test", type=int, default=500000, help="Maximum number of pair test to perform. Use random sampling to speed up above this number.")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)
    logger = logging.getLogger("MOT")

    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()

    mystery_word_main(args.dictionary_file, args.num_letters, args.compute_best_words, args.strategy, args.max_pair_test, logger)

    if args.profile:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumulative')
        stats.print_stats(20)
