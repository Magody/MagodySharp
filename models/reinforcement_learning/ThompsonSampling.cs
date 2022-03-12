using System;
using System.Collections;
using System.Collections.Generic;

using types;
using custom_lib;


namespace reinforcement_learning
{
    public class ThompsonSampling
    {
        
        private double[][] X;
        private static Random randomGenerator = new Random();
        private static RandomBeta betavariate = new RandomBeta();


        public ThompsonSampling(double[][] X){
            // X - reward_matrix - nxm - n experiments/customers, m strategies/actions
            // each experiment can have multiple reward, since a custome could like multiple ads.
            this.X = X;
        }

        public Hashtable selectBest(bool return_gaining=true){
            Hashtable output_best = new Hashtable();

            int n = this.X.Length, m = this.X[0].Length;


            VectorI strategies_selected_rs = new VectorI();
            double total_reward_rs = 0;

            VectorI strategies_selected_ts = new VectorI();
            VectorD histogram_ts = VectorD.zeros(m);
            double total_reward_ts = 0;
            VectorI numbers_of_rewards_1 = VectorI.zeros(m);
            VectorI numbers_of_rewards_0 = VectorI.zeros(m);
            VectorD rewards_strategies = VectorD.zeros(m);
            VectorD regret = new VectorD();


            
            for (var i = 0; i < n; i++)
            {
                // Random Strategy
                if(return_gaining){
                    int strategy_rs = randomGenerator.Next(0,m);
                    strategies_selected_rs.Add(strategy_rs);
                    double reward_rs = this.X[i][strategy_rs];
                    total_reward_rs = total_reward_rs + reward_rs;
                }
                // Thompson Sampling
                int strategy_ts = 0;
                double max_random = 0;

                for (var j = 0; j < m; j++)
                {
                    double random_beta = betavariate.sample(numbers_of_rewards_1[j] + 1, numbers_of_rewards_0[j] + 1);
                    if(random_beta > max_random){
                        max_random = random_beta;
                        strategy_ts = j;
                    }
                }
                double reward_ts = X[i][strategy_ts];

                if(reward_ts == 1){
                    numbers_of_rewards_1[strategy_ts] = numbers_of_rewards_1[strategy_ts] + 1;
                    histogram_ts[strategy_ts] += 1;
                }
                else{
                    numbers_of_rewards_0[strategy_ts] = numbers_of_rewards_0[strategy_ts] + 1;
                }
                    
               
                strategies_selected_ts.Add(strategy_ts);
                total_reward_ts = total_reward_ts + reward_ts;
                // Best Strategy
                double total_reward_bs = double.MinValue;
                for (var j = 0; j < m; j++)
                {
                    rewards_strategies[j] = rewards_strategies[j] + X[i][j];
                    total_reward_bs = Math.Max(total_reward_bs,rewards_strategies[j]);
                }
                // Regret
                regret.Add(total_reward_bs - total_reward_ts);
            }

            double return_absolute = total_reward_ts - total_reward_rs;
            double return_relative = (total_reward_ts - total_reward_rs) / total_reward_rs * 100;
            

            int best_strategy = 0;
            double best_strategy_max_random = 0;

            for (var j = 0; j < m; j++)
            {
                double random_beta = betavariate.sample(numbers_of_rewards_1[j] + 1, numbers_of_rewards_0[j] + 1);
                if(random_beta > best_strategy_max_random){
                    best_strategy_max_random = random_beta;
                    best_strategy = j;
                }
            }


            for (var i = 0; i < histogram_ts.Count; i++)
            {
                histogram_ts[i] = (double)histogram_ts[i] / n;
            }

            output_best["return_absolute"] = return_absolute;
            output_best["return_relative"] = return_relative;
            output_best["best_strategy"] = best_strategy;
            output_best["regret"] = regret;
            output_best["histogram"] = histogram_ts;
            

            return output_best;
        }
        
           
        

        
    }
}
