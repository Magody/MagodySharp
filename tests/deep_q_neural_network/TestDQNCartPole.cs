using System;
using System.Collections.Generic;
using System.Collections;

using types;
using custom_lib;
using reinforcement_learning;
using neural_network;
using layers;
using loss;

using test;


using System.Text.Json;




namespace magodysharp
{
    

    class TestDQNCartPole
    {
        
        public static void tryTest()
        {

             Random randomGenerator = new Random();


            ////////////////////MODEL/////////////////
            // HYPERPARAMETERS: vars and functions
           
            int epochs = 10; // epochs are inside neural networks
            float learning_rate = 0.001f;
            int batch_size = 32;
            FunctionLossEnum loss = FunctionLossEnum.mse;
            float gamma = 0.99f;
            float epsilon = 1f;
            int total_episodes = 100;
            float target_learning_rate = 0.001f;
            float decay_rate_alpha = (float)(((float)learning_rate/target_learning_rate)-1)/(float)(total_episodes);
            int interval_for_learning = 1; // (int)(EnvironmentDataCenter.timeday_in_minutes/3f);
            int interval_for_update_models = 25; // (int)(EnvironmentDataCenter.timemonth_in_minutes/100f);

            


            // HYPERPARAMETERS: Topology
            int number_inputs = 4;
            int number_actions = 2;  // 0 left or 1 right
            Sequential sequentialNetwork = new Sequential(new List<Layer>{
                new Dense(3,DenseInitializationMode.kaiming_he,number_inputs),
                new Activation(ActivationFunctionEnum.relu),
                new Dense(3,DenseInitializationMode.kaiming_he),
                new Activation(ActivationFunctionEnum.relu),
                new Dense(number_actions,DenseInitializationMode.xavier)
            });

            // Setup
            EnvironmentCartPole environmentCartPole = new EnvironmentCartPole("CartPole1", TypeExecution.train);

            NNConfig nnConfig = new NNConfig(epochs,learning_rate,batch_size,loss,0,decay_rate_alpha);
            QLearningConfig qLearningConfig = new QLearningConfig(
                gamma,epsilon,total_episodes,interval_for_learning, interval_for_update_models,
                experience_replay_reserved_space:100
            );

            QNeuralNetwork qNeuralNetwork = new QNeuralNetwork(
                nnConfig,
                qLearningConfig,
                null,
                sequentialNetwork
            );       

            // CALLBACK: Episode: Begin
            Agent.CallbackOnEpisodeBegin onEpisodeBegin = (int episode, EnvironmentRL env) => { };

            Agent.CallbackOnCollectObservations onCollectObservations = (int episode, EnvironmentRL env) => {
                return env.getState().parseToFloatArray(); 
            };

            Agent.CallbackOnActionPerformed onActionPerformed = (int episode, EnvironmentRL env, float[] action_values) => {
                int action = QLearning.selectActionHigher(action_values);
                Hashtable context = new Hashtable{{"force", EnvironmentCartPole.discrete_actuator_force(action)}};
                env.step(context);
            };

            QLearning.FunctionReward functionRewardCartPole = (QLearning self, VectorD state, int action_selected, EnvironmentRL env) => {
                ReplayTuple replayTuple = new ReplayTuple();
                float reward = 0;
                VectorD new_state = new VectorD();

                EnvironmentCartPole environment = env as EnvironmentCartPole;
                // Stop if the network fails to keep the cart within the position or angle limits.
                bool game_over_due_position = MathF.Abs(environment.x) >= environment.position_limit;
                bool game_over_due_angle = MathF.Abs(environment.theta) >= environment.angle_limit_radians;
                bool is_terminal = game_over_due_position || game_over_due_angle;

                
                // reward = environment.angle_limit_radians - MathF.Abs(environment.theta);

                /*
                if(environment.theta > 0){
                    reward = (action_selected == 0)? -1:1;
                }
                else if(environment.theta < 0){
                    reward = (action_selected == 1)? -1:1;
                }

                
                */

                if(is_terminal){
                    reward = -10;
                }
                else{
                    reward = 1;
                }

                
                
                replayTuple.state = state; // old state where action was selected
                replayTuple.action = action_selected;
                replayTuple.reward = reward;
                replayTuple.new_state = environment.getState(); // already updated in OnActionPerformed
                replayTuple.is_terminal = is_terminal;

                string output = "s:[";
                foreach (double item in replayTuple.state)
                {
                    output += $"{Math.Round(item, 3)},";
                }
                string direction = (replayTuple.action == 0)? "LEFT":"RIGHT";
                output += $"], a:{direction}, r:{replayTuple.reward}, s':[";
                foreach (double item in replayTuple.new_state)
                {
                    output += $"{Math.Round(item, 3)},";
                }
                output += $"], angle {environment.theta}";

                if(is_terminal && game_over_due_position){
                    Console.WriteLine("POSITION ERROR");
                   // Console.WriteLine(output); 
                }


                return replayTuple;
            };

            Agent.CallbackOnEpisodeEnd onEpisodeEnd = (int episode, EnvironmentRL env) => {  
                // SAVING THE MODEL
                if(env.reward > env.reward_best_episode){
                    env.reward_best_episode = env.reward;
                    qNeuralNetwork.save($"./dump/{env.name}.json");  
                } 
            
                Console.WriteLine($"Episode {episode}: reward {env.reward}. steps {(int)(MathF.Round(env.timestep,3)/0.01f)}. cost {MathF.Round(env.cost,6)}");
                    
                
            };

            // every step is 0.01 seconds, the episode ends after this counter reach 60
            AgentDQNN agent01 = new AgentDQNN(
                qNeuralNetwork, onEpisodeBegin, onCollectObservations, onActionPerformed,
                onEpisodeEnd, decision_period:0, maximum_steps:(int)environmentCartPole.simulation_time_seconds
            );


            agent01.runEpisodesAsync(environmentCartPole, qNeuralNetwork.qLearningConfig.total_episodes, functionRewardCartPole);


        }
    }

    
}
