
import random
import config as cfg
import sarsa


class ExpectedSarsa(sarsa.Sarsa):
    """
    Expected Sarsa:
        Q(s, a) += alpha * (reward(s,a) + gamma * expected_value(Q(s', a')) - Q(s,a))

        * alpha is the learning rate.
        * gamma is the value of the future reward.
    It uses the expected value of the next state-action pair to update the former state.
    """
    def __init__(self, actions, alpha=cfg.alpha, gamma=cfg.gamma, epsilon=cfg.epsilon):
        super().__init__(actions, alpha, gamma, epsilon)

    # learn
    def learn(self, state1, action, state2, reward):
        old_utility = self.q.get((state1, action), None)
        if old_utility is None:
            self.q[(state1, action)] = reward

        # update utility
        else:
            expected_value = sum([self.get_utility(state2, a) for a in self.actions]) / len(self.actions)
            self.q[(state1, action)] = old_utility + self.alpha * (reward + self.gamma * expected_value - old_utility)