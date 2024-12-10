# -*- coding: utf-8 -*-
"""

.. moduleauthor:: Valentin Emiya
"""

import numpy as np
import matplotlib.pyplot as plt

class LinearBandits:
    """
    Linear bandit problem

    Parameters
    ----------
    n_arms : int
        Number of arms or actions
    n_features : int
        Number of features
    """

    def __init__(self, n_arms, n_features):
        self._theta = np.random.randn(n_features, n_arms)

    @property
    def n_arms(self):
        return self._theta.shape[1]

    @property
    def n_features(self):
        return self._theta.shape[0]

    def step(self, a, x):
        """
        Parameters
        ----------
        a : int
            Index of action/arm
        x : ndarray
            Context (1D array)

        Returns
        -------
        float
            Reward
        """

        assert 0 <= a
        assert a < self.n_arms
        return np.vdot(x, self._theta[:, a]) + np.random.randn()

    def get_context(self):
        """
        Returns
        -------
        ndarray
            Context (1D array)
        """
        return np.random.randn(self.n_features)

    def __str__(self):
        return '{}-arms linear bandit in dimension {}'.format(self.n_arms,
                                                              self.n_features)


class LinUCBAlgorithm:
    def __init__(self, n_arms, n_features, delta):
        self.alpha = 1 + np.sqrt(0.5 * np.log(2 / delta))
        self.A = np.array([np.identity(n_features) for _ in range(n_arms)])
        self.b = np.zeros((n_features, n_arms))
        self.theta = np.zeros((n_features, n_arms))

    @property
    def n_arms(self):
        return self.A.shape[0]

    @property
    def n_features(self):
        return self.A.shape[1]

    def get_action(self, x):
        ucb_values = []
        for k in range(self.n_arms):
            theta_k = np.linalg.solve(self.A[k], self.b[:, k])
            x_A_inv_x = np.dot(x.T, np.linalg.solve(self.A[k], x))
            ucb = np.dot(x, theta_k) + self.alpha * np.sqrt(x_A_inv_x)
            ucb_values.append(ucb)
        return np.argmax(ucb_values)

    def fit_step(self, action, reward, x):
        self.A[action] += np.outer(x, x)
        self.b[:, action] += reward * x

def simulate_bandits(n_arms=30, n_features=10, n_iterations=1000, delta=0.1):
    env = LinearBandits(n_arms, n_features)
    agent = LinUCBAlgorithm(n_arms, n_features, delta)

    regrets = []
    cumulative_regret = 0

    for t in range(n_iterations):
        x = env.get_context()
        action = agent.get_action(x)
        reward = env.step(action, x)
        optimal_reward = max([np.dot(x, env._theta[:, k]) for k in range(n_arms)])
        regret = optimal_reward - reward
        cumulative_regret += regret
        regrets.append(cumulative_regret)
        agent.fit_step(action, reward, x)

    return regrets


n_arms = 30
n_features = 10
n_iterations = 1000

regrets = simulate_bandits(n_arms=n_arms, n_features=n_features, n_iterations=n_iterations)

plt.figure(figsize=(10, 6))
plt.plot(regrets, label="Regret cumulatif")
plt.xlabel("Itérations")
plt.ylabel("Regret cumulatif")
plt.title("Regret cumulatif de LinUCB")
plt.legend()
plt.grid()
plt.show()

#--------------------------------------------#!SECTION "Optimisation de l'algorithme LinUCB"--------------------------------------------

import time

class LinUCBAlgorithmOptimized:
    def __init__(self, n_arms, n_features, delta):
        self.alpha = 1 + np.sqrt(0.5 * np.log(2 / delta))
        self.A_inv = np.array([np.identity(n_features) for _ in range(n_arms)])
        self.b = np.zeros((n_features, n_arms))
        self.theta = np.zeros((n_features, n_arms))

    def get_action(self, x):
        ucb_values = []
        for k in range(len(self.A_inv)):
            theta_k = np.dot(self.A_inv[k], self.b[:, k])
            x_A_inv_x = np.dot(x.T, np.dot(self.A_inv[k], x))
            ucb = np.dot(x, theta_k) + self.alpha * np.sqrt(x_A_inv_x)
            ucb_values.append(ucb)
        return np.argmax(ucb_values)

    def fit_step(self, action, reward, x):
        A_inv = self.A_inv[action]
        x_outer = np.outer(x, x)
        x_A_inv = np.dot(A_inv, x)
        factor = 1.0 / (1.0 + np.dot(x.T, x_A_inv))

        # Mise à jour de rang 1
        self.A_inv[action] -= factor * np.outer(x_A_inv, x_A_inv)
        self.b[:, action] += reward * x


# Comparaison des performances
def compare_algorithms(n_arms=30, n_features=10, n_iterations=1000, delta=0.1):
    env = LinearBandits(n_arms, n_features)

    # Version standard
    standard_agent = LinUCBAlgorithm(n_arms, n_features, delta)
    start_time_standard = time.time()
    for _ in range(n_iterations):
        x = env.get_context()
        action = standard_agent.get_action(x)
        reward = env.step(action, x)
        standard_agent.fit_step(action, reward, x)
    time_standard = time.time() - start_time_standard

    # Version optimisée
    optimized_agent = LinUCBAlgorithmOptimized(n_arms, n_features, delta)
    start_time_optimized = time.time()
    for _ in range(n_iterations):
        x = env.get_context()
        action = optimized_agent.get_action(x)
        reward = env.step(action, x)
        optimized_agent.fit_step(action, reward, x)
    time_optimized = time.time() - start_time_optimized

    return time_standard, time_optimized



time_standard, time_optimized = compare_algorithms()

print(time_standard,time_optimized)
#5.097937107086182 0.11358022689819336