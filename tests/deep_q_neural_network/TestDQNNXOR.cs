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
    

    class TestDQNNXOR
    {
        
        public static void tryTest()
        {
            ////////////////////HYPERPARAMETERS/////////////////
            // HYPERPARAMETERS: vars and functions
            Random randomGenerator = new Random();

            int epochs = 5; // epochs are inside neural networks
            float learning_rate = 0.001f;
            int batch_size = 32;
            FunctionLossEnum loss = FunctionLossEnum.mse;
            float gamma = 0.9f;
            float epsilon = 1f;
            int total_episodes = 100;
            float target_learning_rate = 0.001f;
            float decay_rate_alpha = (float)(((float)learning_rate/target_learning_rate)-1)/(float)(total_episodes);
            int interval_for_learning = 5; // (int)(EnvironmentDataCenter.timeday_in_minutes/3f);
            int interval_for_update_models = interval_for_learning*2; // (int)(EnvironmentDataCenter.timemonth_in_minutes/100f);

            


            // HYPERPARAMETERS: Topology
            int number_inputs = 2;
            int number_actions = 2;
            Sequential sequentialNetwork = new Sequential(new List<Layer>{
                new Dense(3,DenseInitializationMode.kaiming_he,number_inputs),
                new Activation(ActivationFunctionEnum.relu),
                new Dense(number_actions,DenseInitializationMode.xavier)
            });

            // Setup
            EnvironmentCommon environmentXOR = new EnvironmentCommon("XOR");

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

            // Global variables        

            // CALLBACK: Episode: Begin
            Agent.CallbackOnEpisodeBegin onEpisodeBegin = (int episode, EnvironmentRL env) => { };

            Agent.CallbackOnCollectObservations onCollectObservations = (int episode, EnvironmentRL env) => {
                float[] state = new float[2];
                for (int i = 0; i < 2; i++)
                {
                    state[i] = (float)randomGenerator.Next(0,2);
                }
                return state;            
            };

            Agent.CallbackOnActionPerformed onActionPerformed = (int episode, EnvironmentRL env, float[] action_values) => { };

            QLearning.FunctionReward functionRewardXOR = (QLearning self, VectorD state, int action_selected, EnvironmentRL env) => {
                ReplayTuple replayTuple = new ReplayTuple();
                float reward = 0;
                VectorD new_state = new VectorD();
                bool is_terminal = false; // after this state there are no more states.

                EnvironmentCommon environment = env as EnvironmentCommon;

                int a = (int)state[0];
                int b = (int)state[1];
                if(a == b){
                    reward = (action_selected == 0)? 1:-1;
                }
                else{
                    reward = (action_selected == 1)? 1:-1;
                }
                replayTuple.state = state;
                replayTuple.action = action_selected;
                replayTuple.reward = reward;
                replayTuple.new_state = VectorD.clone(state);
                replayTuple.is_terminal = is_terminal;

                return replayTuple;
            };

            Agent.CallbackOnEpisodeEnd onEpisodeEnd = (int episode, EnvironmentRL env) => {  
                // SAVING THE MODEL
                if(env.reward > env.reward_best_episode){
                    env.reward_best_episode = env.reward;
                    qNeuralNetwork.save($"./dump/{env.name}.json");  
                }  
                Console.WriteLine($"Episode {episode}: reward {env.reward}.");
            };


            AgentDQNN agent01 = new AgentDQNN(
                qNeuralNetwork, onEpisodeBegin, onCollectObservations, onActionPerformed,
                onEpisodeEnd, decision_period:0, maximum_steps:200
            );


            agent01.runEpisodesAsync(environmentXOR, qNeuralNetwork.qLearningConfig.total_episodes, functionRewardXOR);


            // Test
            environmentXOR.setTypeExecution(TypeExecution.test);
            List<VectorD> statesTest = new List<VectorD>();
            statesTest.Add(new VectorD{0,0});
            statesTest.Add(new VectorD{0,1});
            statesTest.Add(new VectorD{1,0});
            statesTest.Add(new VectorD{1,1});

            foreach (VectorD state in statesTest)
            {
                QSelection qSelection = qNeuralNetwork.selectAction(state, environmentXOR.is_validation_or_test);
                int action = qSelection.action_index;
                ReplayTuple newReplayTuple = functionRewardXOR(qNeuralNetwork, state, action, environmentXOR);
                Console.WriteLine(state + " -> Reward: " + newReplayTuple.reward);
            }


            // deepQLearningDataCenter();


        }
    }

    
}
