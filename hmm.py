from typing import Optional
import numpy as np


class HMM():

    def __init__(self, n_hidden: int, n_obs: int) -> None:
        self.n_hidden = n_hidden
        self.n_obs = n_obs
        self.transition_matrix = np.eye(n_hidden)
        self.emission_matrix = np.ones((n_obs, n_hidden))
        self.emission_matrix /= self.emission_matrix.sum(axis=0)
        self.initial_state = np.ones(n_hidden) / n_hidden
        self.state = self.initial_state
    

    def sample(self):
        state_probs = self.transition_matrix @ self.state
        state_indx = np.random.choice(self.n_hidden, p=state_probs)
        self.state = np.eye(self.n_hidden)[state_indx, :]
        obs_probs = self.emission_matrix @ self.state
        obs_indx = np.random.choice(self.n_obs, p=obs_probs)
        # obs = np.eye(self.n_obs)[obs_indx, :]
        return obs_indx

    def forward(self, obs_indx):
        prior = self.transition_matrix @ self.initial_state
        likelihood = np.diag(self.emission_matrix[obs_indx, :])
        unnorm_state = likelihood @ self.transition_matrix @ self.initial_state
        self.state = unnorm_state / unnorm_state.sum()
    
    def em():
        pass

    def reset(self):
        self.state = self.initial_state



if __name__ == "__main__":

    hmm = HMM(3, 2)
    data = []
    for i in range(10):
        data.append(hmm.sample())
    print(data)

    hmm.reset()
    for datum in data:
        hmm.forward(datum)
        print(hmm.state)