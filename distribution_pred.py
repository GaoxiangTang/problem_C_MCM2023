# %%
# import the optimized strategy wrote by 3b1b
from wordle.simulations import *


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %%
df = pd.read_excel("pdata.xlsx")

# %%
from scipy.special import softmax

def optimal_guesses(allowed_words, possible_words, priors,
                  look_two_ahead=False,
                  optimize_for_uniform_distribution=False,
                  purely_maximize_information=True,
                  strategy="optimized",
                  max_trial=5,
                  ):
    if len(possible_words) == 1:
        return [(possible_words[0], 1)]
    if strategy == "entropy-maximization":
        weights = get_weights(possible_words, priors)
        preference = get_entropies(allowed_words, possible_words, weights)
    if strategy == "optimized":
        preference = -get_score_lower_bounds(
            allowed_words, possible_words
        )
    if strategy == "frequency-oriented":
        preference = get_weights(allowed_words, priors)
    trial_times = min(len(possible_words), max_trial)
    top_idx = np.argsort(preference)[-trial_times:]
    guesses = [allowed_words[i] for i in top_idx]
    top_pref = np.array([preference[i] for i in top_idx])
    print(softmax(top_pref))
    return list(zip(guesses, softmax(top_pref)))

    # Just experimenting here...
    if optimize_for_uniform_distribution:
        expected_scores = get_score_lower_bounds(
            allowed_words, possible_words
        )
    else:
        expected_scores = get_expected_scores(
            allowed_words, possible_words, priors,
            look_two_ahead=look_two_ahead
        )
    return allowed_words[np.argmin(expected_scores)]

# %%
from collections import defaultdict
import pickle 

def simulate_games_human(first_guess=None,
                   priors=None,
                   look_two_ahead=False,
                   optimize_for_uniform_distribution=False,
                   exclude_seen_words=False,
                   test_set=None,
                   shuffle=False,
                   hard_mode=False,
                   purely_maximize_information=True,
                   results_file=None,
                   quiet=False,
                   strategy="optimized",
                   ):
    all_words = get_word_list(short=False)
    short_word_list = get_word_list(short=True)

    if priors is None:
        priors = get_frequency_based_priors()

    if test_set is None:
        test_set = short_word_list

    if shuffle:
        random.shuffle(test_set)

    seen = set()
    next_guess_map = {}

    # Function for choosing the next guess, with a dict to cache
    # and reuse results that are seen multiple times in the sim

    def get_next_guesses(possibilities, strategy, phash):
        if phash in next_guess_map:
            return next_guess_map[phash]
        choices = possibilities if hard_mode else all_words
        next_guess_map[phash] = optimal_guesses(
                choices, possibilities, priors,
                strategy=strategy,
                look_two_ahead=look_two_ahead,
                purely_maximize_information=purely_maximize_information,
                optimize_for_uniform_distribution=optimize_for_uniform_distribution,
            )
        return next_guess_map[phash]
    
    class freq_dist():

        def __init__(self, dist=None) -> None:
            if dist is None:
                self.dist = defaultdict(lambda: 0)
            else:
                self.dist = dist

        def merge(self, other, prob):
            for freq, w in other.dist.items():
                self.dist[freq + 1] += w * prob

    def distribution_prediction(possibilities, strategy="optimized", phash=""):
        dist = freq_dist()
        # print(guesses)
        for guess, prob in get_next_guesses(possibilities, strategy, phash):
            if guess == answer:
                dist.merge(freq_dist({0: 1}), prob)
                continue
            pattern = get_pattern(guess, answer)
            new_phash = phash + str(guess) + "".join(map(str, pattern_to_int_list(pattern)))
            dist.merge(distribution_prediction(get_possible_words(guess, pattern, possibilities), strategy, new_phash), prob)
        
        return dist

    # Go through each answer in the test set, play the game,
    # and keep track of the stats.
    for answer in ProgressDisplay(test_set, leave=False, desc=" Trying all wordle answers"):
        
        possibilities = list(filter(lambda w: priors[w] > 0, all_words))

        if exclude_seen_words:
            possibilities = list(filter(lambda w: w not in seen, possibilities))
        patterns = []
        guesses = []
        print(answer)
        print(distribution_prediction(possibilities, strategy).dist)

    # with open('next_guess_map.pkl', 'wb') as f:
    #     pickle.dump(dictionary, f)
        
    # with open('saved_dictionary.pkl', 'rb') as f:
    #     loaded_dict = pickle.load(f)
        


# %%
results, decision_map = simulate_games_human(
        test_set=df["Word"].to_list(),
        priors=get_true_wordle_prior(),
        optimize_for_uniform_distribution=True,
        strategy="optimized"
    )

# %%



