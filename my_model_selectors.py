import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        state_score = float('inf')
        best_score = float('inf')

        best_model = self.base_model(self.n_constant)

        for component_state in range(self.min_n_components, self.max_n_components+1):
            curr_model = None    
            try:
                curr_model = self.base_model(component_state)
                logL = curr_model.score(self.X, self.lengths)
                p = component_state ** 2 + 2 * component_state * model.n_features -1
                state_score = -2 * logL + p * math.log(component_state)
            except:
                pass
            if state_score < best_score : 
                best_score, best_model = state_score, curr_model

        return best_model



class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        best_model = self.base_model(self.n_constant)
        best_score = float('-inf')
        
        dictionary = list(self.words)
        dictionary.remove(self.this_word)
        for component_state in range(self.min_n_components, self.max_n_components+1):
            try:
                curr_model = self.base_model(component_state)
                logL = curr_model.score(self.X, self.lengths)
                logL_other_sum = 0

                for word in dictionary: 
                    X, lengths = self.hwords[word]
                    logL_word = curr_model.score(X, lengths)
                    logL_other_sum += logL_word
                curr_score = logL - logL_other_sum/(len(dictionary)-1)
            
                if curr_score > best_score : 
                    best_model, best_score = curr_model, score
            except: 
                pass
            
        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        kfold_splits = 4
        best_score = float('-inf')
        best_model = self.base_model(self.n_constant)

        for component_state in range(self.min_n_components, self.max_n_components+1):
            mean_scores = []
            curr_model = None
            
            if len(self.sequences) < kfold_splits:
                break
            split_method = KFold(kfold_splits, random_state=self.random_state, shuffle=False)
            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                X_train, lengths_train = combine_sequences(cv_train_idx, self.sequences)
                X_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)
                try:
                    curr_model = GaussianHMM(n_components=component_state, n_iter=1000).fit(X_train, lengths_train)
                    logL = curr_model.score(X_test, lengths_test)
                    mean_scores.append(logL)
                except:
                    pass
            if len(mean_scores) > 0 : 
                avg_score = np.average(mean_scores)
            else :
                avg_score = float('-inf')
            if avg_score > best_score : 
                best_score, best_model = avg_score, curr_model

        return best_model

