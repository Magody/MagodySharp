using System;
using System.Collections;
using System.Collections.Generic;

using System.IO;

using types;
using custom_lib;

using System.Text.Json;

namespace reinforcement_learning
{
    public struct ReplayTuple
    {
        public VectorD state {get;set;}
        public int action {get;set;}
        public float reward {get;set;}
        public VectorD new_state {get;set;}
        public bool is_terminal {get;set;}
    }

    public enum EpsilonUpdate
    {
        interpolation
    }
    public struct QSelection
    {
        public double max_q;
        public VectorD q;
        public int action_index;
    }

    public enum GameReplayStrategy
    {
        linear, cache
    }

    public class QLearningConfig{

        public int total_episodes {get;set;}
        public int total_episodes_test {get;set;}
        public float gamma {get;set;}
        public float initial_epsilon {get;set;}
        public GameReplayStrategy gameReplayStrategy {get;set;}
        public int experience_replay_reserved_space {get;set;}  // space per class

        public int interval_for_learning {get;set;}
        public int interval_for_update_models {get;set;}


        

        public Dictionary<string,float> rewards;


        public QLearningConfig(
            float gamma, 
            float epsilon, 
            int total_episodes,
            int interval_for_learning,
            int interval_for_update_models = 1,
            int experience_replay_reserved_space = 100,
            GameReplayStrategy gameReplayStrategy = GameReplayStrategy.cache,
            int total_episodes_test = 1
        ){
            this.gamma = gamma;
            this.initial_epsilon = epsilon;
            this.total_episodes = total_episodes;
            this.gameReplayStrategy = gameReplayStrategy;
            this.experience_replay_reserved_space = experience_replay_reserved_space;
            this.interval_for_learning = interval_for_learning;
            this.interval_for_update_models = interval_for_update_models;
            this.total_episodes_test = total_episodes_test;

            rewards = new Dictionary<string, float>{
                {"correct",1f}, {"incorrect",-1f}, {"nothing",0f}
            };

        }
        public Hashtable getJSONHash(){

            Hashtable hash = new Hashtable();
            hash["total_episodes"] = this.total_episodes;
            hash["total_episodes_test"] = this.total_episodes_test;
            hash["gamma"] = this.gamma;
            hash["initial_epsilon"] = this.initial_epsilon;
            hash["gameReplayStrategy"] = $"{this.gameReplayStrategy}";
            hash["experience_replay_reserved_space"] = this.experience_replay_reserved_space;
            hash["interval_for_learning"] = this.interval_for_learning;
            hash["interval_for_update_models"] = this.interval_for_update_models;

            return hash;
        }

        public static QLearningConfig buildQLearningConfig(JsonElement element){

            string gameReplayStrategy_string = element.GetProperty("gameReplayStrategy").GetString();
            GameReplayStrategy gameReplayStrategy = GameReplayStrategy.cache;

            switch (gameReplayStrategy_string)
            {
                case "linear":
                    gameReplayStrategy = GameReplayStrategy.linear;
                    break;
                case "cache":
                    gameReplayStrategy = GameReplayStrategy.cache;
                    break;
            }


            QLearningConfig qLearningConfig = new QLearningConfig(
                (float)element.GetProperty("gamma").GetDouble(),
                (float)element.GetProperty("initial_epsilon").GetDouble(),
                element.GetProperty("total_episodes").GetInt32(),
                element.GetProperty("interval_for_learning").GetInt32(),
                element.GetProperty("interval_for_update_models").GetInt32(),
                element.GetProperty("experience_replay_reserved_space").GetInt32(),
                gameReplayStrategy,
                element.GetProperty("total_episodes_test").GetInt32()
            );

            return qLearningConfig;

        }
    }

    public class QLearning
    {
        public bool useCustomRunEpisodes {get; set;}


        // should return: reward, new_state, finish
        public delegate ReplayTuple FunctionReward(QLearning self, VectorD state, int action_selected, EnvironmentRL environment);
        public delegate Hashtable FunctionRunEpisodes(QLearning self, FunctionReward functionReward, EnvironmentRL environment);
        public delegate Hashtable FunctionEpisode(QLearning self, int episode, FunctionReward functionReward, EnvironmentRL environment);

        public FunctionRunEpisodes functionRunEpisodes {get;set;}
        public FunctionEpisode functionEpisode {get;set;}

        // Config
        public QLearningConfig qLearningConfig {get;set;}
            
        // Game replay
        public ReplayTuple?[] gameReplay {get;set;}
        public bool[] gameReplayUsage {get;set;}
        protected int gameReplayCounter {get;set;}
        public int gameReplayNumValidElements {get;set;}
        
        
        // aux
        public int[] index_action {get;set;}
        
        // epsilon decay
        public float epsilon {get;set;}

        public static Random randomGenerator = new Random();

        public QLearning(){

        }

        public QLearning(
            QLearningConfig qLearningConfig
        ){
            // maleable base for other types of episodes
            this.qLearningConfig = qLearningConfig;
            this.epsilon = this.qLearningConfig.initial_epsilon;
            this.useCustomRunEpisodes = false;
        }
        
        public QLearning(
            QLearningConfig qLearningConfig,
            FunctionEpisode functionEpisode
        ){
            this.gameReplayCounter = 0;
            this.qLearningConfig = qLearningConfig;
            this.epsilon = this.qLearningConfig.initial_epsilon;
            this.functionEpisode = functionEpisode;
            this.useCustomRunEpisodes = false;
        }

        public void initGameReplay(int actions_length){
            this.index_action = new int[actions_length];
            int size_literal = this.qLearningConfig.experience_replay_reserved_space * actions_length;

            this.gameReplay = new ReplayTuple?[size_literal];
            this.gameReplayUsage = new bool[size_literal];
            for (var i = 0; i < size_literal; i++)
            {
                this.gameReplayUsage[i] = false;
                this.gameReplay[i] = null;
            }
            this.gameReplayNumValidElements = 0;
        }

        public ReplayTuple[] getGameReplayBatch(int batch_size, bool randomize=true){
            // todo: optimize this
            List<ReplayTuple> gameReplayValid = new List<ReplayTuple>();
            
            for (var i = 0; i < this.gameReplay.Length; i++)
            {
                if(this.gameReplayUsage[i]){
                    ReplayTuple replayTuple = (ReplayTuple)this.gameReplay[i];
                    gameReplayValid.Add(QLearning.cloneReplayTuple(replayTuple));
                    
                }
            }

            ReplayTuple[] gameReplayBatch = new ReplayTuple[batch_size];

            for (int i = 0; i < batch_size; i++)
            {
                int index = randomGenerator.Next(0, gameReplayValid.Count);
                gameReplayBatch[i] = gameReplayValid[index];
                gameReplayValid.RemoveAt(index);
            }

            return gameReplayBatch;

        }



        public static ReplayTuple cloneReplayTuple(ReplayTuple target){
            ReplayTuple replayTuple = new ReplayTuple();
            replayTuple.state = VectorD.clone(target.state);
            replayTuple.action = target.action;
            replayTuple.reward = target.reward;
            replayTuple.new_state = VectorD.clone(target.new_state);
            replayTuple.is_terminal = target.is_terminal;
            return replayTuple;
        }
        
        public void transferGameReplay(ReplayTuple[] gameReplayOther){
            int gameReplayLength = gameReplayOther.Length;
            int actions_length = gameReplayLength / this.qLearningConfig.experience_replay_reserved_space;
            this.initGameReplay(actions_length);

            for (var i = 0; i < gameReplayLength; i++)
            {
                this.gameReplay[i] = QLearning.cloneReplayTuple(gameReplayOther[i]);
                this.gameReplayUsage[i] = true;
                this.gameReplayNumValidElements += 1;              
            }
        }

        public void setCustomRunEpisodes(FunctionRunEpisodes functionRunEpisodes){
            this.useCustomRunEpisodes = true;
            this.functionRunEpisodes = functionRunEpisodes;
        }

        public Hashtable runEpisodes(FunctionReward functionReward, EnvironmentRL environment){
            if(this.useCustomRunEpisodes){
                return this.functionRunEpisodes(this, functionReward, environment);
            }
            else{
                return this.runEpisodesDefault(this, functionReward, environment);
            }
        }

        public Hashtable runEpisodesDefault(QLearning self, FunctionReward functionReward, EnvironmentRL environment){
            
            bool is_test = environment.is_validation_or_test;
            int verbose_level = environment.verbose_level;

            Hashtable history_episodes = new Hashtable();

            int total_episodes = this.qLearningConfig.total_episodes;
            if(is_test)
                total_episodes = this.qLearningConfig.total_episodes_test;
            

            float[] history_rewards = new float[total_episodes];
            List<VectorD> history_update_costs = new List<VectorD>();

            int debug_step = Math.Max(1,total_episodes/100);

            for (var episode = 1; episode <= total_episodes; episode++)
            {
                        
                
                Hashtable history_episode = this.functionEpisode(self, episode, functionReward, environment);


                
                history_rewards[episode-1] = (float)history_episode["reward_cummulated"];
                
                VectorD update_costs = (VectorD)history_episode["update_costs"];
                history_update_costs.Add(update_costs);
                string extra = (string)history_episode["extra"];
                double cost_last = -1;
                if(update_costs.Count > 0){
                    cost_last = VectorD.mean(update_costs);
                }

                if(verbose_level >= 1 && (episode%debug_step==0 || episode == 1 || episode == total_episodes)){

                    Console.WriteLine($"Episode {episode}: Cost {cost_last}: reward {history_rewards[episode-1]}. Step: {environment.timestep}. {extra}. Best r {environment.reward_best_episode}"); // , Epsilon: {this.epsilon}");
                
                    // Console.WriteLine($"Episode {episode+1} of {total_episodes}, test?={is_test}");
                } 


                bool early_stopping = (bool)history_episode["early_stopping"];
                if(early_stopping){
                    Console.WriteLine("EARLY STOP");
                    Console.WriteLine($"Episode {episode}: Cost {cost_last}: reward {history_rewards[episode-1]}. Step: {environment.timestep}. {extra}. Best r {environment.reward_best_episode}"); // , Epsilon: {this.epsilon}");
                    break;
                }
                

            } 

            history_episodes.Add("rewards", history_rewards);
            history_episodes.Add("update_costs", history_update_costs);

            return history_episodes;
                        
        }

        public void saveReplayTuple(ReplayTuple replayTuple){
            int index_replay = -1;
            switch (this.qLearningConfig.gameReplayStrategy)
            {
                case GameReplayStrategy.linear:
                    index_replay = this.gameReplayCounter % this.gameReplay.Length;
                    break;
                case GameReplayStrategy.cache:
                    int index_experience_replay = this.index_action[replayTuple.action] % this.qLearningConfig.experience_replay_reserved_space;
                    this.index_action[replayTuple.action] += 1;
                    
                    int offset = (replayTuple.action) * this.qLearningConfig.experience_replay_reserved_space;
                    index_replay = offset+index_experience_replay;
                    break;
                default:
                    throw new Exception("Unexpected GameReplayStrategy");
            }
            this.gameReplayCounter = this.gameReplayCounter + 1;

            if(!this.gameReplayUsage[index_replay]){
                // new object
                this.gameReplayNumValidElements += 1;
            }

            this.gameReplayUsage[index_replay] = true;
            this.gameReplay[index_replay] = QLearning.cloneReplayTuple(replayTuple);
        }


        public void updateEpsilonDecay(int episode, EpsilonUpdate epsilonUpdate=EpsilonUpdate.interpolation){
            switch (epsilonUpdate)
            {
                case EpsilonUpdate.interpolation:
                    double frac = Math.Log(Math.Exp(1) - ((Math.Exp(1) - 1) * ((double)episode/this.qLearningConfig.total_episodes)));
                    this.epsilon = (float) Math.Max(0.01, this.qLearningConfig.initial_epsilon * frac);
                    break;
                default:
                    throw new Exception("Unknown epsilon update");
            }
            
        }

        public static QSelection selectActionQEpsilonGreedy(VectorD Qval, float epsilon, int num_actions, bool is_test){
            
            Dictionary<string,double> res_max = VectorD.max(Qval);
            double max_q = res_max["value"];
            int idx = (int)res_max["index"];

            QSelection selection = new QSelection();

            if(!is_test){
                double rng = QLearning.randomGenerator.NextDouble();
                // epsilon-greedy
                if(rng <= epsilon){
                    int[] full_action_list = new int[num_actions-1];
                    int index_list = 0;
                    for (var a = 0; a < num_actions; a++)
                    {
                        if(a != idx){
                            full_action_list[index_list++] = a;
                        }
                        
                    }
                    
                    int idx_valid_action = randomGenerator.Next(0,full_action_list.Length);
                    idx = full_action_list[idx_valid_action];
                }
            }
            
            selection.max_q = max_q;
            selection.action_index = idx;
            selection.q = Qval;

            return selection;
        }

        public virtual Hashtable getJSONHash(){

            Hashtable hash = new Hashtable();
            return hash;
        }


        public virtual string save(string path_file_name="temp.json", bool append = false){
            return "{}";
        }

        public static QLearning load(string path_file_name="temp.json"){
            return new QLearning();
        }


        public static int selectActionHigher(float[] actions_values){
            float maximum_value = float.MinValue;
            int action = -1;

            for (int i = 0; i < actions_values.Length; i++)
            {
                if(actions_values[i] > maximum_value){
                    maximum_value = actions_values[i];
                    action = i;
                }
            }
            
            return action;
        }

                

            
    }
}
