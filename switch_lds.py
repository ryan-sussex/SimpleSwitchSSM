"""
Following https://www.cs.toronto.edu/~hinton/absps/switch.pdf
"""
from hmm import HMM
from kalman_filter import KalmanFilter

class SwitchedStateSpace():
    
    def __init__(
            self,
            n_switches: int,
            n_hidden: int,
            n_obs: int
    ):
        self.n_switches = n_switches
        self.n_obs = n_obs
        self.n_hidden = n_hidden
        
        # We don't need the observation part of this model
        self.switching_model = HMM(self.n_switches, self.n_switches)
        self.state_space_models = dict()
        for switch in range(n_switches):
            self.state_space_models[switch] = KalmanFilter(
                n_obs,
                n_hidden
            )
    
    def sample(self, verbose:bool = True):
        switch = self.switching_model.sample(update_state=True)
        
        lds_samples = dict()
        for (m, lds) in self.state_space_models.items():
            lds_samples[m] = lds.sample()

        if verbose:
            print(f"switch state: {switch}") 
        
        return lds_samples[switch]
    


if __name__ == "__main__":


    ref_switch_ssm = SwitchedStateSpace(n_switches=2, n_hidden=1, n_obs=1)

    traj = []
    for i in range(10):
        traj.append(ref_switch_ssm.sample())
    
    print(traj)