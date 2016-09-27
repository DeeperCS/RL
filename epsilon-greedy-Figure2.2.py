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
        action_record = np.zeros(runs)
        for s in range(runs):
            # pick action 
            action = self.pick_action()

            # obtain reward with noise
            reward_record[s] = self.rewards[action] + np.random.normal(0, 1)
            # record optimal action
            action_record[s] = 1 if action == np.argmax(self.rewards) else 0
            
            # updating the action value for action choosen
            self.q_values[action] = self.q_values[action] + (1.0/(self.counts[action]+1))*(self.rewards[action] - self.q_values[action])
            
            # count ++
            self.counts[action] += 1
        return reward_record, action_record
    
########    ########   ########   ########   ########   ########   
runTimes = 2000
taskSteps = 1000
rewards1 = np.zeros(taskSteps)
actions1 = np.zeros(taskSteps)

rewards2 = np.zeros(taskSteps)
actions2 = np.zeros(taskSteps)

rewards3 = np.zeros(taskSteps)
actions3 = np.zeros(taskSteps)

for i in range(runTimes):
    q = np.random.normal(0, 1, 10)
    
    agent = SampleAverage(0.01, q)
    (r, a) = agent.calc_rewards(taskSteps)
    rewards1 += r
    actions1 += a
    
    agent = SampleAverage(0.1, q)
    (r, a) = agent.calc_rewards(taskSteps)
    rewards2 += r
    actions2 += a
    
    agent = SampleAverage(0, q)
    (r, a) = agent.calc_rewards(taskSteps)
    rewards3 += r
    actions3 += a

###########Average Reward##########    
plt.figure()
plt.plot(rewards2/runTimes, 'g', label='$\epsilon=0.1$')
plt.plot(rewards1/runTimes, 'r', label='$\epsilon=0.01$')
plt.plot(rewards3/runTimes, 'b', label='$\epsilon=0$')

plt.legend(loc=4)
plt.xlabel('Steps')
plt.ylabel('Average Rewards')
plt.xlim(-25, 1000)
plt.savefig('AverateReward.png')
# plt.close('all')

###########Optial Action##########
plt.figure()
plt.plot(actions2/runTimes, 'g', label='$\epsilon=0.1$')
plt.plot(actions1/runTimes, 'r', label='$\epsilon=0.01$')
plt.plot(actions3/runTimes, 'b', label='$\epsilon=0$')

plt.legend(loc=4)
plt.xlabel('Steps')
plt.ylabel('Optimal Action')
plt.xlim(-25, 1000)
plt.savefig('OptimalAction.png')