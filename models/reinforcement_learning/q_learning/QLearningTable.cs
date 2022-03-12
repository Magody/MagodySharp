using System;
using System.Collections;
using System.Collections.Generic;

using types;
using custom_lib;


namespace reinforcement_learning
{
    public class QLearningTable:QLearning
    {

        public QLearningTable(
            QLearningConfig qLearningConfig
        ):base(qLearningConfig){
            this.functionEpisode = this.runEpisodeQTable;
            this.setCustomRunEpisodes(this.customRunEpisodesQTable);
        }
        
        public QLearningTable(
            QLearningConfig qLearningConfig,
            FunctionEpisode functionEpisode
        ):base(qLearningConfig,functionEpisode){
            this.functionEpisode = functionEpisode;
            this.setCustomRunEpisodes(this.customRunEpisodesQTable);
        }

        public Hashtable customRunEpisodesQTable(QLearning self, QLearning.FunctionReward functionReward, EnvironmentRL environment){
            

            bool is_test = environment.is_validation_or_test;
            int verbose_level = environment.verbose_level;

            int num_nodes = (int)environment.context["num_actions"];
            MatrixD Q_table = MatrixD.zeros(num_nodes,num_nodes);
            
            Hashtable history_episodes = new Hashtable();
            
            VectorD history_q_val = new VectorD();
            VectorD history_scores = new VectorD();

            Random randomGenerator = new Random();

            for (var episode = 0; episode < this.qLearningConfig.total_episodes; episode++)
            {
                // rand world
                VectorD state = new VectorD{(double)randomGenerator.Next(0, num_nodes)};
                environment.context["state"] = state;
                environment.context["Q_table"] = Q_table;
                // Console.WriteLine(Q_table);
            
                Hashtable history_episode = this.functionEpisode(self, episode, functionReward, environment);
                int action = (int)history_episode["action"];
                double update = (double)history_episode["update"];
                
                Q_table[(int)state[0]][action] = update;
                    
                history_q_val.Add((double)history_episode["max_q"]);
                history_scores.Add((double)history_episode["score"]);

            }
                
            
            history_episodes["history_q_val"] = history_q_val;
            history_episodes["Q_table"] = Q_table;
            history_episodes["history_scores"] = history_scores;

            return history_episodes;
        }
    


        public Hashtable runEpisodeQTable(QLearning self, int episode, QLearning.FunctionReward functionReward, EnvironmentRL environment){

                Hashtable history_episode =new Hashtable();


                bool is_test = environment.is_validation_or_test;
                int verbose_level = environment.verbose_level;

    
                VectorD state = (VectorD)environment.context["state"];
                MatrixD Q_table = (MatrixD)environment.context["Q_table"];
                int num_actions = (int)environment.context["num_actions"];
                
                VectorD q_values_actual = Q_table[(int)state[0]];

                QSelection q_a = QLearning.selectActionQEpsilonGreedy(q_values_actual,this.epsilon, num_actions, is_test);
                int action = q_a.action_index;

                // context is modified by reference
                ReplayTuple outputReward = functionReward(self,state,action,environment);
                double reward = outputReward.reward;
                VectorD new_state = outputReward.new_state;
                bool finish = outputReward.is_terminal;
                // Console.WriteLine($"epsilon {qLearning.epsilon} -> {(int)state[0]}->{action}->{(int)new_state[0]} = {reward}");
                VectorD q_values_next = Q_table[(int)new_state[0]];
                double max_q = VectorD.max(q_values_next)["value"];
                
                double score = 0;


                double max_in_q_table= double.MinValue;
                for (var i = 0; i < Q_table.Count; i++)
                {
                    for (var j = 0; j < Q_table[0].Count; j++)
                    {
                        if(Q_table[i][j] > max_in_q_table){
                            max_in_q_table = Q_table[i][j];
                        }
                    }
                }

                if(max_q > 0)
                    score = (Q_table/max_in_q_table).sum(3)[0][0];  // normalization


                // learn
                double update = 0;
                if(finish)
                    update = reward;
                else
                    update = reward + this.qLearningConfig.gamma * max_q;
                
                
                history_episode["reward"] = reward;
                history_episode["update"] = update;
                history_episode["action"] = action;
                history_episode["max_q"] = max_q;
                history_episode["score"] = score;
                
                this.updateEpsilonDecay(episode);

                return history_episode;
                
            }
            
    }
}
