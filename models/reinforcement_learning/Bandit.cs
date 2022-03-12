using System;
using System.Collections;
using System.Collections.Generic;

using types;
using custom_lib;


namespace reinforcement_learning
{
    public class Bandit
    {
        private float p = 0f;
        public float p_estimate {get;set;}
        private int N = 0;
        private float old_estimate = 0f;
        private Random randomGenerator;

        public Bandit(float p){
            this.p_estimate = 0;
            this.p = p;
            this.randomGenerator = new Random();
        }
        
        public int pull(){
            return (randomGenerator.NextDouble() < (double)this.p)? 1:0;
        }

        public void update(int reward){
            int x = reward;
            this.N = this.N + 1;
            this.p_estimate = this.old_estimate + ((float)1/this.N) * (x - this.old_estimate);
            this.old_estimate = this.p_estimate;
        }   


        public static string kArmedBanditDemoSimulation(){
            string output = "";
            int NUM_TRIALS = 10000;
            float EPS = 0.2f;
            // for non stationary bandits we should use  a learning rate, because N
            // becoming inf means that new data is irrelevant for algorithm
            float[] BANDIT_PROBABILITIES = new float[]{0.2f, 0.5f, 0.75f};

            int total_bandits = BANDIT_PROBABILITIES.Length;

            List<Bandit> bandits = new List<Bandit>();

            float max_p = float.MinValue;

            for (var i = 0; i < total_bandits; i++)
            {
                float p = BANDIT_PROBABILITIES[i];
                bandits.Add(new Bandit(p));
            }
            
            List<int> rewards = new List<int>();

            int num_times_explored = 0;
            int num_times_exploited = 0;
            int num_optimal = 0;

            Random random = new Random();

            for (var trial = 0; trial < NUM_TRIALS; trial++)
            {
                int actual_bandit;
                if(random.NextDouble() < EPS){
                    // explore
                    num_times_explored = num_times_explored + 1;
                    actual_bandit = random.Next(0, total_bandits);
                }
                else{
                    // exploit
                    num_times_exploited = num_times_exploited + 1;
                    
                    max_p = float.MinValue;
                    actual_bandit = -1;  // first assumptiom if i dont know the probabilities
                    
                    for (var i = 0; i < total_bandits; i++)
                    {
                        float p = bandits[i].p_estimate;
                        if(p > max_p){
                            max_p = p;
                            actual_bandit = i;
                        }
                    }
                }
                // hard coded because i know the third is the best
                if(actual_bandit == 2)
                    num_optimal = num_optimal + 1;
                
                int reward = bandits[actual_bandit].pull();
                rewards.Add(reward);
                bandits[actual_bandit].update(reward);
                // bandits(actual_bandit)

            }

            // Console.WriteLine mean estimates for each bandit
            int c = 1;
            foreach (var b in bandits)
            {
                output += $"{c++} mean estimate: {b.p_estimate}\n";
            }

            float rewards_sum = 0;
            foreach (var item in rewards)
            {
                rewards_sum += item;
            }
                

            // Console.WriteLine total reward
            output += $"total reward earned: {rewards_sum}\n";
            output += $"overall win rate: {rewards_sum/ NUM_TRIALS}\n";
            output += $"num_times_explored: {num_times_explored}\n";
            output += $"num_times_exploited: {num_times_exploited}\n";
            output += $"num times selected optimal bandit: {num_optimal}\n";
            
            return output;

        }     
        

        
    }
}
