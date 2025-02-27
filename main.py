import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

__version__ = "2.0.1"

class NumberGuessingEnv(gym.Env):
    def __init__(self, lower_bound=1, upper_bound=100, answer=None):
        super(NumberGuessingEnv, self).__init__()
        
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        if answer is None:
            self.answer = random.randint(lower_bound, upper_bound)
        else:
            self.answer = answer
            
        self.current_guess = None
        self.num_guesses = 0
        self.max_guesses = 20
        
        
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, -1]), 
            high=np.array([1, 1, 1, 1]), 
            dtype=np.float32
        )
        
        self.generator_action_space = spaces.Discrete(11) 
        
        self.validator_action_space = spaces.Discrete(2)
        
        self.current_low = lower_bound
        self.current_high = upper_bound
        self.last_result = 0  
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_low = self.lower_bound
        self.current_high = self.upper_bound
        self.num_guesses = 0
        self.last_result = 0
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        norm_low = (self.current_low - self.lower_bound) / (self.upper_bound - self.lower_bound)
        norm_high = (self.current_high - self.lower_bound) / (self.upper_bound - self.lower_bound)
        norm_guesses = self.num_guesses / self.max_guesses
        
        return np.array([norm_low, norm_high, norm_guesses, self.last_result], dtype=np.float32)
    
    def generator_step(self, action):
        """
        Process GENERATOR's action (making a guess)
        
        Args:
            action: int from 0-10 representing percentile between current bounds
            
        Returns:
            New state for VALIDATOR
        """
        self.num_guesses += 1
        
        if action == 0:
            self.current_guess = (self.current_low + self.current_high) // 2
        else:
            percentile = action / 10.0
            self.current_guess = int(self.current_low + percentile * (self.current_high - self.current_low))
        
        self.current_guess = max(self.current_low, min(self.current_high, self.current_guess))
        
        return self._get_observation()
    
    def validator_step(self, action=None):
        """
        Process VALIDATOR's action (confirming or denying the guess)
        In this implementation, the validator is deterministic based on the answer
        
        Args:
            action: Not used in this implementation since validation is deterministic
                   Could be used in extensions where validation is learned
                   
        Returns:
            observation, reward, done, truncated, info
        """
        if self.current_guess == self.answer:
            self.last_result = 0.5
            reward = 1.0
            done = True
        elif self.current_guess < self.answer:
            self.last_result = -1
            reward = 0.0
            self.current_low = self.current_guess + 1
            done = self.num_guesses >= self.max_guesses
        else:  
            self.last_result = 1 
            reward = 0.0
            self.current_high = self.current_guess - 1
            done = self.num_guesses >= self.max_guesses
            
        truncated = False
        
        info = {
            "answer": self.answer,
            "guess": self.current_guess,
            "distance": abs(self.current_guess - self.answer),
            "num_guesses": self.num_guesses
        }
        
        return self._get_observation(), reward, done, truncated, info


class GeneratorAgent:
    """The agent responsible for generating guesses"""
    
    def __init__(self, action_space_size, state_space_size, learning_rate=0.01, discount_factor=0.95):
        self.q_table = np.zeros((state_space_size, action_space_size))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = 1.0
        self.min_exploration_rate = 0.01
        self.exploration_decay = 0.005
        self.action_space_size = action_space_size
        
    def state_to_index(self, state):
        """Convert continuous state to discrete indices for Q-table"""
        low_idx = min(9, int(state[0] * 10))
        high_idx = min(9, int(state[1] * 10))
        guess_idx = min(9, int(state[2] * 10))
        
        if state[3] == -1:
            result_idx = 0
        elif state[3] == 0.5:
            result_idx = 1
        else: 
            result_idx = 2
            
        state_index = low_idx + high_idx * 10 + guess_idx * 100 + result_idx * 1000
        return min(state_index, self.q_table.shape[0] - 1)
    
    def choose_action(self, state):
        """Select action using epsilon-greedy policy"""
        if random.random() < self.exploration_rate:
            return random.randint(0, self.action_space_size - 1)
        else:
            state_idx = self.state_to_index(state)
            return np.argmax(self.q_table[state_idx])
    
    def learn(self, state, action, reward, next_state, done):
        """Update Q-values using Q-learning algorithm"""
        state_idx = self.state_to_index(state)
        next_state_idx = self.state_to_index(next_state)
        
        if done:
            target = reward
        else:
            target = reward + self.discount_factor * np.max(self.q_table[next_state_idx])
        
        self.q_table[state_idx][action] += self.learning_rate * (target - self.q_table[state_idx][action])
    
    def update_exploration_rate(self, episode, total_episodes):
        """Decrease exploration rate over time"""
        self.exploration_rate = self.min_exploration_rate + \
            (1.0 - self.min_exploration_rate) * np.exp(-self.exploration_decay * episode)


def train(lower_bound=1, upper_bound=100, episodes=1000, answers=None):
    """Train the generator agent over multiple episodes"""
    
    state_bins = 100  
    
    generator = GeneratorAgent(
        action_space_size=11,  
        state_space_size=state_bins
    )
    
    total_rewards = []
    guess_counts = []
    
    for episode in range(episodes):
        answer = answers[episode % len(answers)] if answers is not None else None
        
        env = NumberGuessingEnv(lower_bound=lower_bound, upper_bound=upper_bound, answer=answer)
        
        state, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            generator_action = generator.choose_action(state)
            validator_state = env.generator_step(generator_action)
            next_state, reward, done, _, info = env.validator_step()
            generator.learn(state, generator_action, reward, next_state, done)
            state = next_state
            episode_reward += reward
        
        generator.update_exploration_rate(episode, episodes)
        
        total_rewards.append(episode_reward)
        guess_counts.append(info["num_guesses"])
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(total_rewards[-100:])
            avg_guesses = np.mean(guess_counts[-100:])
            print(f"Episode {episode+1}/{episodes} - Avg Return: {avg_reward:.2f}, Avg Guesses: {avg_guesses:.2f}")
            print(f"q_table: {generator.q_table}")
    
    return generator, total_rewards, guess_counts


def test_agent(generator, lower_bound=1, upper_bound=100, test_episodes=100, answers=None):
    """Test the trained generator against new problems"""
    guess_counts = []
    success_rate = 0
    
    for episode in range(test_episodes):
        answer = answers[episode % len(answers)] if answers is not None else None
        
        env = NumberGuessingEnv(lower_bound=lower_bound, upper_bound=upper_bound, answer=answer)
        
        state, _ = env.reset()
        done = False
        
        while not done:
            generator.exploration_rate = 0
            generator_action = generator.choose_action(state)
            
            validator_state = env.generator_step(generator_action)
            
            next_state, reward, done, _, info = env.validator_step()
            
            state = next_state
        
        guess_counts.append(info["num_guesses"])
        if info["guess"] == env.answer:
            success_rate += 1
    
    success_rate /= test_episodes
    avg_guesses = np.mean(guess_counts)
    
    print(f"Test Results:")
    print(f"Success Rate: {success_rate*100:.2f}%")
    print(f"Average Guesses: {avg_guesses:.2f}")
    print(f"Optimal Guesses (Binary Search): {int(np.log2(upper_bound - lower_bound + 1))}")
    
    return success_rate, avg_guesses



if __name__ == "__main__":
    LOWER_BOUND = 1
    UPPER_BOUND = 5
    TRAINING_EPISODES = 25000
    
    ANSWERS = None
    
    print("Training generator agent...")
    generator, rewards, guesses = train(
        lower_bound=LOWER_BOUND, 
        upper_bound=UPPER_BOUND,
        episodes=TRAINING_EPISODES,
        answers=ANSWERS
    )
    
    print("\nTesting generator agent...")
    test_agent(
        generator,
        lower_bound=LOWER_BOUND,
        upper_bound=UPPER_BOUND,
        test_episodes=100,
        answers=ANSWERS
    )