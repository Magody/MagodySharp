using System;
using System.Collections;
using System.Collections.Generic;

using types;
using custom_lib;

using reinforcement_learning;

namespace test
{
    public class EnvironmentDataCenter:EnvironmentRL
    {

        public float[] monthly_atmospheric_temperatures {get;set;}
        public int initial_month {get;set;}
        public float atmospheric_temperature {get;set;}
        public float[] optimal_temperature {get;set;}
        public float min_temperature {get;set;}
        public float max_temperature {get;set;}
        public int min_number_users {get;set;}
        public int max_number_users {get;set;}
        public int max_update_users {get;set;}
        public float min_rate_data {get;set;}
        public int max_rate_data {get;set;}
        public int max_update_data {get;set;}
        public int initial_number_users {get;set;}
        public int current_number_users {get;set;}
        public float initial_rate_data {get;set;}
        public float current_rate_data {get;set;}
        public float intrinsic_temperature {get;set;}
        public float temperature_ai {get;set;}
        public float temperature_noai {get;set;}
        public float total_energy_ai {get;set;}
        public float total_energy_noai {get;set;}
        public float temperature_step {get;set;}
        public float direction_boundary {get;set;}
        public static int timeday_in_minutes = 24 * 60;
        public static int timemonth_in_minutes = 30 * timeday_in_minutes;

        
        public EnvironmentDataCenter(
            string name,
            float[] optimal_temperature,
            float direction_boundary,
            int initial_month = 0,
            int initial_number_users = 10,
            int initial_rate_data = 20, 
            TypeExecution typeExecution=TypeExecution.train, int verbose_level=1,
            int patience_max=-1,
            float temperature_step=2f
        ):base(name,typeExecution,patience_max,verbose_level){

            this.temperature_step = temperature_step;
            this.direction_boundary = direction_boundary;
            this.monthly_atmospheric_temperatures = new float[]{1.0f, 5.0f, 7.0f, 10.0f, 11.0f, 20.0f, 23.0f, 24.0f, 22.0f, 10.0f, 5.0f, 1.0f};
            this.initial_month = initial_month;
            this.atmospheric_temperature = this.monthly_atmospheric_temperatures[initial_month];
            this.optimal_temperature = optimal_temperature;
            this.min_temperature = -20;
            this.max_temperature = 80;
            this.min_number_users = 10;
            this.max_number_users = 100;
            this.max_update_users = 5;
            this.min_rate_data = 20;  // GB/TB, etc
            this.max_rate_data = 300;  // GB/TB, etc
            this.max_update_data = 10;
            this.initial_number_users = initial_number_users;
            this.initial_rate_data = initial_rate_data;
            this.reset(new Hashtable{{"new_month",initial_month}});
            
        }

        public override void reset(Hashtable extraInfo){

            int new_month = (int)extraInfo["new_month"];

            this.atmospheric_temperature = this.monthly_atmospheric_temperatures[new_month];
            this.initial_month = new_month;
            this.current_number_users = EnvironmentRL.randomGenerator.Next(this.initial_number_users);// this.initial_number_users;
            this.current_rate_data = EnvironmentRL.randomGenerator.Next((int)this.initial_rate_data);
            this.intrinsic_temperature = this.atmospheric_temperature + 1.25f * this.current_number_users + 1.25f * this.current_rate_data;
            
             // Both (ai,noai) temperatures follow the intrinsic temperature
            this.temperature_ai = this.intrinsic_temperature;
            this.temperature_noai = this.intrinsic_temperature; // (this.optimal_temperature[0] + this.optimal_temperature[1]) / 2.0f;
            this.total_energy_ai = 0.0f;
            this.total_energy_noai = 0.0f;
            this.reward = 0.0f;
            this.game_over = false;

        }

        public override VectorD getState(){
            VectorD state = new VectorD();

            float scaled_temperature_ai = (float)(this.temperature_ai - this.min_temperature) / (float)(this.max_temperature - this.min_temperature);
            float scaled_number_users = (float)(this.current_number_users - this.min_number_users) / (float)(this.max_number_users - this.min_number_users);
            float scaled_rate_data = (float)(this.current_rate_data - this.min_rate_data) / (float)(this.max_rate_data - this.min_rate_data);
            
            state.Add(scaled_temperature_ai);
            state.Add(scaled_number_users);
            state.Add(scaled_rate_data);
            return state;

        }

        public static Hashtable functionEpisode(QLearning self, int episode, QLearning.FunctionReward functionReward, EnvironmentRL env){
                
            Hashtable history = new Hashtable();

            QNeuralNetwork qNeuralNetwork = self as QNeuralNetwork;

            EnvironmentDataCenter environment = env as EnvironmentDataCenter;
            
            qNeuralNetwork.updateEpsilonDecay(episode);

            float total_reward = 0;
            int new_month = QLearning.randomGenerator.Next(0,12);
            environment.reset(new Hashtable{{"new_month", new_month}});


            VectorD update_costs = new VectorD();

            VectorD state = environment.getState();
            environment.timestep = 0;
            // STARTING THE LOOP OVER ALL THE TIMESTEPS (1 Timestep = 1 Minute) IN ONE EPOCH
            // Maximum the episode is for 5 months
            while ((!environment.game_over) && environment.timestep <= 1 * EnvironmentDataCenter.timemonth_in_minutes){
                
                
                QSelection qSelection = qNeuralNetwork.selectAction(state, environment.is_validation_or_test);
                int action = qSelection.action_index;
                ReplayTuple newReplayTuple = functionReward(self, state, action, environment);

                if(episode == qNeuralNetwork.qLearningConfig.total_episodes && newReplayTuple.is_terminal){
                    string summary = $"{Math.Round(newReplayTuple.state[0],2)} -> {newReplayTuple.action} -> {newReplayTuple.reward} -> {Math.Round(newReplayTuple.new_state[0],2)} -> End:{newReplayTuple.is_terminal} -> q: {qSelection.q}";
                    Console.WriteLine($"Step {environment.timestep}: {summary}");

                }


                total_reward += newReplayTuple.reward;
                // STORING THIS NEW TRANSITION INTO THE MEMORY
                if(environment.is_train_only){
                    qNeuralNetwork.saveReplayTuple(newReplayTuple);

                    // train in intervals to optimice
                    if(environment.timestep % qNeuralNetwork.qLearningConfig.interval_for_learning == 0){
                        ExperienceReplayHistoryLearning history_learning = qNeuralNetwork.learnFromExperienceReplay(episode, environment.verbose_level-1);
                        if(history_learning.learned){
                            update_costs.Add(history_learning.mean_cost);
                        }
                    }

                    if(environment.timestep % qNeuralNetwork.qLearningConfig.interval_for_update_models == 0){
                        qNeuralNetwork.updateQNeuralNetworkTarget();
                    }
                }

                environment.timestep += 1;
                state = newReplayTuple.new_state;
                // Sanity check: environment already have game_over = false
                if(newReplayTuple.is_terminal){
                    break;
                }
            }

            if(update_costs.Count == 0 && environment.is_train_only){
                ExperienceReplayHistoryLearning history_learning = qNeuralNetwork.learnFromExperienceReplay(episode, environment.verbose_level-1);
                if(history_learning.learned){
                    update_costs.Add(history_learning.mean_cost);
                }
            }


            history["total_energy_ai"] = environment.total_energy_ai;
            history["total_energy_noai"] = environment.total_energy_noai;
            history["early_stopping"] = environment.isEarlyStopping(total_reward);  

            history["reward_cummulated"] = total_reward;  
            history["extra"] = $". Energy saved: {environment.total_energy_noai-environment.total_energy_ai}";  
            history["update_costs"] = update_costs;

                            
            // SAVING THE MODEL
            if(total_reward > environment.reward_best_episode){
                environment.reward_best_episode = total_reward;
                qNeuralNetwork.save($"{environment.name}.json");  
            }              
            
            return history;
        }

        public static ReplayTuple functionReward(QLearning self, VectorD state, int action_selected, EnvironmentRL env){
            ReplayTuple replayTuple = new ReplayTuple();
            float reward = 0;
            VectorD new_state = new VectorD();
            bool is_terminal = false;

            EnvironmentDataCenter environment = env as EnvironmentDataCenter;


            int month = (int)((float)environment.timestep / EnvironmentDataCenter.timemonth_in_minutes);

            // cooldown/heat that will perform the ai
            float energy_ai = Math.Abs(action_selected - environment.direction_boundary) * environment.temperature_step;

            // GETTING THE REWARD
            // Computing the energy spent by the server’s cooling system when there is no AI
            float energy_noai = 0;
            if(environment.temperature_noai < environment.optimal_temperature[0]){
                energy_noai = environment.optimal_temperature[0] - environment.temperature_noai;
                environment.temperature_noai = environment.optimal_temperature[0];
            }
            else if(environment.temperature_noai > environment.optimal_temperature[1]){
                energy_noai = environment.temperature_noai - environment.optimal_temperature[1];
                environment.temperature_noai = environment.optimal_temperature[1];
            }
            
            float energy_saved = (energy_noai - energy_ai);  // scaling reward

            



            // GETTING THE NEXT STATE
            // Updating the atmospheric temperature
            environment.atmospheric_temperature = environment.monthly_atmospheric_temperatures[month];
            // Updating the number of users
            int add_Users = EnvironmentRL.randomGenerator.Next(-environment.max_update_users, environment.max_update_users+1);
            environment.current_number_users += add_Users;
            
            if(environment.current_number_users > environment.max_number_users)
                environment.current_number_users = environment.max_number_users;
            else if (environment.current_number_users < environment.min_number_users)
                environment.current_number_users = environment.min_number_users;

            // Updating the rate of data
            environment.current_rate_data += EnvironmentRL.randomGenerator.Next(-environment.max_update_data, environment.max_update_data+1);
            if(environment.current_rate_data > environment.max_rate_data)
                environment.current_rate_data = environment.max_rate_data;
            else if (environment.current_rate_data < environment.min_rate_data)
                environment.current_rate_data = environment.min_rate_data;
            // Computing the Delta of Intrinsic Temperature
            float past_intrinsic_temperature = environment.intrinsic_temperature;
            environment.intrinsic_temperature = environment.atmospheric_temperature + 1.25f * environment.current_number_users + 1.25f * environment.current_rate_data;
            float delta_intrinsic_temperature = environment.intrinsic_temperature - past_intrinsic_temperature;
            // Computing the Delta of Temperature caused by the AI
            float mid = (float)(self.index_action.Length-1)/2f;
            float delta_temperature_ai = energy_ai;
            if (action_selected < mid)
                delta_temperature_ai = -energy_ai;
            

            float optimal_middle = (float)(environment.optimal_temperature[0] + environment.optimal_temperature[1])/2f;
                
            float energy_abs = Math.Abs(energy_ai);

            float reward_by_direction = 0f;
            if(delta_temperature_ai > 0){
                if(environment.temperature_ai < optimal_middle){
                    reward_by_direction += Math.Abs(energy_saved)*2;
                }
                else{
                    reward_by_direction += -Math.Abs(energy_saved);
                }
            }
            else if(delta_temperature_ai < 0){
                if(environment.temperature_ai > optimal_middle){
                    reward_by_direction += Math.Abs(energy_saved)*2;
                }
                else{
                    reward_by_direction += -Math.Abs(energy_saved);
                }
            }


            // Updating the new Server’s Temperature when there is the AI
            environment.temperature_ai += delta_intrinsic_temperature + delta_temperature_ai;
            // Updating the new Server’s Temperature when there is no AI
            environment.temperature_noai += delta_intrinsic_temperature;
            // distance with the new temperature
            float reward_by_distance = (float)1/(Math.Abs((environment.temperature_ai-delta_intrinsic_temperature)-optimal_middle)+1);


            reward = reward_by_direction + reward_by_distance;

            reward += (float)energy_saved;

            // GETTING GAME OVER
            if (environment.temperature_ai < environment.min_temperature){
                if (environment.is_train_only){
                    reward -= 2;
                    if(mid == action_selected){
                        reward = -10;
                    }
                    environment.game_over = true;
                    is_terminal = true;
                }else{
                    environment.total_energy_ai += environment.optimal_temperature[0] - environment.temperature_ai;
                    environment.temperature_ai = environment.optimal_temperature[0];
                }
                    
            }
                
            else if (environment.temperature_ai > environment.max_temperature){
                if (environment.is_train_only){
                    reward -= 2;
                    if(mid == action_selected){
                        reward = -10;
                    }
                    environment.game_over = true;
                    is_terminal = true;
                }else{
                    environment.total_energy_ai += environment.temperature_ai - environment.optimal_temperature[1];
                    environment.temperature_ai = environment.optimal_temperature[1];
                }
            }
            
            // UPDATING THE SCORES
            // Updating the Total Energy spent by the AI
            environment.total_energy_ai += energy_ai;
            // Updating the Total Energy spent by the alternative system when there is no AI
            environment.total_energy_noai += energy_noai;
            
            // SCALING THE NEXT STATE
            new_state = environment.getState();
            

            environment.reward += reward;

            // SUMMARY reward

            replayTuple.state = state;
            replayTuple.action = action_selected;
            replayTuple.reward = reward;
            replayTuple.new_state = new_state;
            replayTuple.is_terminal = is_terminal;

            string summary = $"{replayTuple.state} -> {replayTuple.action} -> {replayTuple.reward} -> {replayTuple.new_state} -> End:{replayTuple.is_terminal}";
            // Console.WriteLine(summary);
            return replayTuple;
        }


        

              
           
        

        
    }
}
