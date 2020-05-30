import numpy as np
import gym

""" Value spaces """
positionSpace = np.linspace(-1.2, 0.6, 20)
velocitySpace = np.linspace(-0.07, 0.07, 20)

""" Retrieving affected spaces """
def getState(observation):
    position, velocity = observation
    positionBinary = np.digitize(position, positionSpace)
    velocityBinary = np.digitize(velocity, velocitySpace)
    return (positionBinary, velocityBinary)

""" Greedy Algorithm style Action Selection """
def maxAction(Q, state, actions=[0,1,2]):
    values = np.array([Q[state, a] for a in actions])
    action = np.argmax(values)
    return action

if __name__ == "__main__":
    env = gym.make("MountainCar-v0")
    env._max_episode_steps = 500
    games = 5000
    alpha = 0.1 #learning rate
    gamma = 0.99 #value of future reward
    epsilon = 1.0 #randomness of action selection
    states = []

    for position in range(21): #21 comes from the 20 buckets - see value spaces
        for velocity in range(21):
            states.append((position, velocity))

    Q = {}
    for state in states:
        for action in [0,1,2]:
            Q[state, action] = 0


    score = 0
    totalRewards = np.zeros(games)
    for i in range(games):
        done = False
        actions = []
        observation = env.reset()
        state = getState(observation)
        if i % 500 == 0 and i >0:
            print('Games run:',i,'Score',np.absolute(score),"epsilon", epsilon)
        score = 0

        while not done:
            action = np.random.choice([0,1,2]) if np.random.random() < epsilon \
                    else maxAction(Q, state)
            actions.append(action)
            newObservation, reward, done, info = env.step(action)
            newState = getState(newObservation)
            score += reward
            newAction = maxAction(Q, state)
            #Q learning algorithm  
            Q[state, action] = Q[state, action] + \
                    alpha*(reward + gamma*Q[newState, newAction] - Q[state, action])
            state = newState
        totalRewards[i] = score
        epsilon = epsilon - 2/games if epsilon > 0.01 else 0.01    
    f = open("actions.txt", "w")
    f.write(str(actions))

