import numpy as np  
import matplotlib.pyplot as plt
%matplotlib inline

class SampleAverage:
    def __init__(self, epsilon, rewards):
        self.epsilon = epsilon
        self.counts = np.zeros(len(rewards))    # count of arm picked up during sampling
        self.q_values = np.zeros(len(rewards))    # action value assigned to each arm
        self.rewards = rewards  # reward behind each door
        
    # This function picks the epsilon greedy action
    def pick_action(self):
        if np.random.rand() > (1-self.epsilon):
            return np.random.randint(len(self.q_values))
        else:
            return np.argmax(self.q_values)
    
    # This function calculates the returns the average rewards for a chosen epsilon
    def calc_rewards(self, runs):
        reward_record = np.zeros(runs)
        for s in range(runs):
            # pick action 
            action = self.pick_action()

            # obtain reward with noise
            reward_record[s] = self.rewards[action] + np.random.normal(0, 1, 1)

            # updating the action value for action choosen
            self.q_values[action] = self.q_values[action] + (1.0/(self.counts[action]+1))*(self.rewards[action] - self.q_values[action])
            
            # count ++
            self.counts[action] += 1
        return reward_record
    
########    ########   ########   ########   ########   ########   
runTimes = 2000
taskSteps = 1000
rewards1 = np.zeros(taskSteps)
rewards2 = np.zeros(taskSteps)
rewards3 = np.zeros(taskSteps)

for i in range(runTimes):
    # generate the true reward Q^{\star}
    q = np.random.normal(0, 1, 10)
    
    agent = SampleAverage(0.01, q)
    rewards1 += agent.calc_rewards(taskSteps)
    
    agent = SampleAverage(0.1, q)
    rewards2 += agent.calc_rewards(taskSteps)
    
    agent = SampleAverage(0, q)
    rewards3 += agent.calc_rewards(taskSteps)

    
plt.plot(rewards2/runTimes, 'g', label='$\epsilon=0.1$')
plt.plot(rewards1/runTimes, 'r', label='$\epsilon=0.01$')
plt.plot(rewards3/runTimes, 'b', label='$\epsilon=0$')

plt.legend(loc=4)
plt.xlabel('Step')
plt.ylabel('Average Rewards')
plt.xlim(-25, 1000)