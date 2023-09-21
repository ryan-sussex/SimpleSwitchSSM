from typing import Optional, List
import numpy as np


class HMM():

    def __init__(
            self, 
            n_hidden: int, 
            n_obs: int,
            transition_prior: Optional[np.array] = None,
            emmission_prior: Optional[np.array] = None
        ) -> None:
        self.n_hidden = n_hidden
        self.n_obs = n_obs
        self.transition_prior = np.ones((n_hidden, n_hidden))
        if transition_prior is not None:
            self.transition_prior = transition_prior
        self.transition_matrix = self.transition_prior / np.sum(self.transition_prior, axis=0)
        self.emission_prior = np.ones((n_obs, n_hidden))
        if emmission_prior is not None:
            self.emission_prior = emmission_prior
        self.emission_matrix = self.emission_prior / self.emission_prior.sum(axis=0)
        self.initial_state = np.ones(n_hidden) / n_hidden
        self.initial_state = np.eye(n_hidden)[0,:]
        self.state = self.initial_state

    def sample(self, update_state=True):
        state_probs = self.transition_matrix @ self.state
        state_indx = np.random.choice(self.n_hidden, p=state_probs)
        state = np.eye(self.n_hidden)[state_indx, :]
        if update_state:
            self.state = state
        obs_probs = self.emission_matrix @ state
        obs_indx = np.random.choice(self.n_obs, p=obs_probs)
        #
        #  obs = np.eye(self.n_obs)[obs_indx, :]
        return obs_indx

    def forward(self, obs_indx):
        # prior = self.transition_matrix @ self.
        likelihood = np.diag(self.emission_matrix[obs_indx, :])
        unnorm_state = likelihood @ self.transition_matrix @ self.state
        self.state = unnorm_state / unnorm_state.sum()
        return self.state

    def backwards(self, obs_indx, backwards_state):
        likelihood = np.diag(self.emission_matrix[obs_indx, :])
        unnorm_state = backwards_state @ self.transition_matrix @ likelihood 
        state = unnorm_state / unnorm_state.sum()
        return state


    def em(self, obs_seq: List[np.array], lr=.01):
        """
        Online version of baum-welch
        """
        state_dists = []
        empirical_transitions = []
        empirical_emmissions = []
        prev_state = self.initial_state
        for obs in obs_seq:
            backwards_prob = self.backwards(obs, np.ones(self.n_hidden))

            empirical_transitions.append(
                 np.expand_dims(backwards_prob, 1) @ 
                 np.expand_dims(prev_state, 0) 
                #  * self.transition_matrix
            )
    
            state_dist = self.forward(obs)
            state_dists.append(state_dist)


            empirical_emmissions.append(
                 np.expand_dims(state_dist, 1) @ 
                 np.expand_dims(np.eye(self.n_obs)[obs,:], 0) 
                #  * self.emission_matrix
            )
            prev_state = state_dist
            # print("counts", empirical_transitions[-1])

        n_visits = np.sum(state_dists, axis=0)
        n_visits /= n_visits.sum()
        # print("n visits", n_visits)
        n_transitions = np.sum(empirical_transitions, axis=0) / n_visits
        n_emissions = np.sum(empirical_emmissions, axis=0) / n_visits
        # print("n transitions", n_transitions)
        # print("n emissions", n_emissions)

        self.transition_prior = self.transition_prior + n_transitions * lr
        self.transition_matrix = self.transition_prior / self.transition_prior.sum(axis=0)
        
        self.emission_prior = self.emission_prior + n_emissions * lr
        self.emission_matrix = self.emission_prior / self.emission_prior.sum(axis=0)

    def em_iterations(self, obs_seq: List[np.array], iters=10):
        for _ in range(iters):
            self.em(obs_seq)
            self.reset()

        

    def reset(self):
        self.state = self.initial_state



if __name__ == "__main__":

    ref_hmm = HMM(
        n_hidden=2,
        n_obs=2,
        transition_prior=np.array([[2,20], [1,7]]),
        emmission_prior=np.array([[1,10], [9,3]])
    )
    data = []
    for i in range(100):
        traj = []
        for i in range(10):
            traj.append(ref_hmm.sample())
            # print(traj)
        data.append(traj)
        ref_hmm.reset()

    test_traj = []
    for i in range(100):
        test_traj.append(ref_hmm.sample())
    


    hmm = HMM(
        2, 2,
        transition_prior=np.array([[5,2], [1,7]]),
        emmission_prior=np.array([[1,10], [9,3]])
    )
    # hmm.reset
    for traj in data:
        # for step in traj:
        # hmm.em_iterations(traj, iters=10)
        hmm.reset()

        pred = 0
        error = 0
        for step in test_traj:#
            # print(pred, step)
            error += int(pred != step)
            log_likelihood = 
            hmm.forward(step)
            pred = hmm.sample(update_state=False)

        print("error:", error)
        hmm.reset()

    print(hmm.transition_prior)
    print("transition probs", hmm.transition_matrix)

    print(hmm.emission_prior)
    print("emission probs")
    print(hmm.emission_matrix)