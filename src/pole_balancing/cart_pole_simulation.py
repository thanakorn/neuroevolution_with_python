import gym

def run_cart_pole_simulation(net, num_episodes, max_timesteps, visualize=False):
    env = gym.make('CartPole-v0')
    num_steps = []
    for episode in range(num_episodes):
        observation = env.reset()
        for t in range(max_timesteps):
            if visualize: env.render()
            action = 1 if net.activate(observation)[0] > 0.5 else 0
            observation, reward, done, info = env.step(action)
            if done or t == max_timesteps - 1:
                num_steps.append(t)
                break
    
    env.close()
    return sum(num_steps) / len(num_steps)