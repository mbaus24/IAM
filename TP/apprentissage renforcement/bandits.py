# -*- coding: utf-8 -*-
"""

.. moduleauthor:: Valentin Emiya
"""

import abc
import numpy as np


class BernoulliMultiArmedBanditsEnv:
    """
    Bandit problem with Bernoulli distributions

    Parameters
    ----------
    means : array-like
        True values (expectation of reward) for each arm
    """

    def __init__(self, means):
        self._means = np.array(means)
        assert np.all(0 <= self._means)
        assert np.all(self._means <= 1)

    @property
    def n_arms(self):
        """
        Number of arms

        Returns
        -------
        int
        """
        return self._means.size

    @property
    def _true_values(self):
        return self._means

    def step(self, a):
        """
        Play an arm and return reward

        Parameters
        ----------
        a : int
            Index of arm to be played

        Returns
        -------
        bool
            Reward obtained from playing arm `a` (true if win, false otherwise)
        """
        assert 0 <= a
        assert a < self.n_arms
        return np.random.rand() < self._means[a]

    def __str__(self):
        return "{}-arms bandit problem with Bernoulli distributions".format(self.n_arms)


class NormalMultiArmedBanditsEnv:
    """
    Bandit problem with normal distributions with unit variance.

    Parameters
    ----------
    means : array-like
        Mean values for each arm
    stds : array-like
        Standard deviation values for each arm (1 if None)
    """

    def __init__(self, means, stds=None):
        self._means = np.array(means)
        if stds is None:
            stds = np.ones_like(means)
        self._stds = stds

    @property
    def n_arms(self):
        """
        Number of arms

        Returns
        -------
        int
        """
        return self._means.size

    @property
    def _true_values(self):
        return self._means

    def step(self, a):
        """
        Play an arm and return reward

        Parameters
        ----------
        a : int
            Index of arm to be played

        Returns
        -------
        float
            Reward obtained from playing arm `a`
        """
        assert 0 <= a
        assert a < self.n_arms
        return np.random.randn() * self._stds[a] + self._means[a]

    def __str__(self):
        return "{}-arms bandit problem with Normal distributions".format(self.n_arms)

    @staticmethod
    def create_random(n_arms):
        """

        Parameters
        ----------
        n_arms : int
            Number of arms or actions

        Returns
        -------

        """
        return NormalMultiArmedBanditsEnv(means=np.random.randn(n_arms))


class BanditAgent(abc.ABC):
    """
    A generic abstract class for Bandit Agents

    Parameters
    ----------
    n_arms : int
        Number of arms
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, n_arms=10):
        self.n_arms = n_arms

    @abc.abstractmethod
    def get_action(self):
        """
        Choose an action (abstract)

        Returns
        -------
        int
            The chosen action
        """
        raise NotImplementedError

    @abc.abstractmethod
    def fit_step(self, action, reward):
        """
        update current value estimates with an (action, reward) pair (abstract)

        Parameters
        ----------
        action : int
        reward : float

        """
        raise NotImplementedError


class RandomBanditAgent(BanditAgent):
    """
    Bandit agent with random actions (pure exploration)

    Parameters
    ----------
    n_arms : int
        Number of arms
    """

    def __init__(self, n_arms=10):
        BanditAgent.__init__(self, n_arms=n_arms)
        # Estimation of the value of each arm
        self._value_estimates = np.zeros(n_arms)
        # Number of times each arm has been chosen
        self._n_estimates = np.zeros(n_arms)

    def get_action(self):
        """
        Choose an action at random uniformly among the available arms

        Returns
        -------
        int
            The chosen action
        """
        return np.random.randint(self.n_arms)

    def fit_step(self, action, reward):
        """
        Do nothing since actions are chosen at random

        Parameters
        ----------
        action : int
        reward : float

        """
        pass


class GreedyBanditAgent(BanditAgent):
    """
    Greedy Bandit Agent (pure exploitation)

    Parameters
    ----------
    n_arms : int
        Number of arms
    """

    def __init__(self, n_arms=10):
        BanditAgent.__init__(self, n_arms=n_arms)
        # Estimation of the value of each arm
        self._value_estimates = np.zeros(n_arms)
        # Number of times each arm has been chosen
        self._n_estimates = np.zeros(n_arms, dtype=int)
        self.initialized = False

    def get_action(self):
        """
        Choose the action with maximum estimated value

        Returns
        -------
        int
            The chosen action
        """
        if not self.initialized:
            for arm in range(self.n_arms):
                if self._n_estimates[arm] == 0:
                    return arm
            self.initialized = True
       
        return np.argmax(self._value_estimates)
        

    def fit_step(self, action, reward):
        """
        update current value estimates with an (action, reward) pair

        Parameters
        ----------
        action : int
        reward : float

        """
        self._n_estimates[action] += 1
        self._value_estimates[action] += (reward - self._value_estimates[action]) / self._n_estimates[action]
        
        return self._value_estimates[action]

        


class EpsilonGreedyBanditAgent(GreedyBanditAgent, RandomBanditAgent):
    """
    Epsilon-greedy Bandit Algorithm

    Parameters
    ----------
    n_arms : int
        Number of arms
    epsilon : float
        Probability to choose an action at random
    """

    def __init__(self, n_arms=10, epsilon=0.1):
        GreedyBanditAgent.__init__(self, n_arms=n_arms)
        self.epsilon = epsilon

    def get_action(self):
        """
        Get Epsilon-greedy action

        Choose an action at random with probability epsilon and a greedy
        action otherwise.

        Returns
        -------
        int
            The chosen action
        """
        return np.random.randint(self.n_arms) if np.random.rand() < self.epsilon else super().get_action()
        

    def fit_step(self, action, reward):
        """
        update current value estimates with an (action, reward) pair

        Parameters
        ----------
        action : int
        reward : float

        """
        return super().fit_step(action, reward)
        


class UcbBanditAgent(GreedyBanditAgent):
    """

    Parameters
    ----------
    n_arms : int
        Number of arms
    c : float
        Positive parameter to adjust exploration/explotation UCB criterion
    """

    def __init__(self, n_arms, c):
        GreedyBanditAgent.__init__(self, n_arms=n_arms)
        self.c = c

    def get_action(self):
        """
        Get UCB action

        Returns
        -------
        int
            The chosen action
        """
        return np.argmax(self._value_estimates + self.c * np.sqrt(np.log(np.sum(self._n_estimates)) / (self._n_estimates + 1)))
        

    def get_upper_confidence_bound(self):
        
        return self._value_estimates + self.c * np.sqrt(np.log(np.sum(self._n_estimates)) / (self._n_estimates + 1))
        

    def fit_step(self, action, reward): 
        """
        update current value estimates with an (action, reward) pair

        Parameters
        ----------
        action : int
        reward : float

        """
        return super().fit_step(action, reward)
        


class ThompsonSamplingAgent(BanditAgent):
    """

    Parameters
    ----------
    n_arms : int
        Number of arms
    """

    def __init__(self, n_arms):
        BanditAgent.__init__(self, n_arms=n_arms)
        self._alpha = np.ones(n_arms)
        self._beta = np.ones(n_arms)
        pass

    def get_action(self):
        
            
        
        return np.argmax(np.random.beta(self._alpha, self._beta))

    def fit_step(self, action, reward):
        self._alpha[action] += reward
        self._beta[action] += 1 - reward
        return self._alpha[action] / (self._alpha[action] + self._beta[action])
    
    def getalpha(self):
        return self._alpha
    def getbeta(self):
        return self._beta
    
