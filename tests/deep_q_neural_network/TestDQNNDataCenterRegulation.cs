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
    

    class TestDQNNDataCenterRegulation
    {
        
        public static void deepQLearningDataCenter(){
            int epochs = 5; // epochs are inside neural networks
            float learning_rate = 0.001f;
            int batch_size = 16;
            FunctionLossEnum loss = FunctionLossEnum.mse;

            float gamma = 0.99f;
            float epsilon = 1f;
            int total_episodes = 100000;
            
            float target_learning_rate = 0.001f;
            float decay_rate_alpha = (float)(((float)learning_rate/target_learning_rate)-1)/(float)(total_episodes);

            int interval_for_learning = 5; // (int)(EnvironmentDataCenter.timeday_in_minutes/3f);
            int interval_for_update_models = interval_for_learning*2; // (int)(EnvironmentDataCenter.timemonth_in_minutes/100f);

            int number_actions = 5;

            float decision_boundary = ((float)number_actions - 1) / 2f;
            EnvironmentDataCenter environment = new EnvironmentDataCenter(
                "testDataCenter", new float[]{18.0f, 24.0f}, decision_boundary, 0, 20, 25
            );

            Sequential sequentialNetwork = new Sequential(new List<Layer>{
                new Dense(8,DenseInitializationMode.kaiming_he,3),
                new Activation(ActivationFunctionEnum.relu),
                new Dropout(0.2f),
                new Dense(8,DenseInitializationMode.kaiming_he),
                new Activation(ActivationFunctionEnum.relu),
                new Dropout(0.1f),
                new Dense(number_actions,DenseInitializationMode.xavier)
            });

            NNConfig nnConfig = new NNConfig(epochs,learning_rate,batch_size,loss,0,decay_rate_alpha);
            QLearningConfig qLearningConfig = new QLearningConfig(
                gamma,epsilon,total_episodes,interval_for_learning, interval_for_update_models,
                experience_replay_reserved_space:100
            );
            QNeuralNetwork qNeuralNetwork = new QNeuralNetwork(nnConfig,qLearningConfig,null,sequentialNetwork);

            qNeuralNetwork.functionEpisode = EnvironmentDataCenter.functionEpisode;

            QLearning.FunctionReward functionReward = EnvironmentDataCenter.functionReward;
            
            Hashtable history = qNeuralNetwork.runEpisodes(functionReward, environment);

            Random random = new Random();
            /*
            for (var i = 0; i < 20; i++)
            {
                ReplayTuple replayTuple = (ReplayTuple)qNeuralNetwork.gameReplay[random.Next(qNeuralNetwork.gameReplay.Length)];
                
                if(!replayTuple.Equals(null)){
                    string summary = $"{Math.Round(replayTuple.state[0],2)} -> {replayTuple.action} -> {replayTuple.reward} -> {Math.Round(replayTuple.new_state[0],2)} -> End:{replayTuple.is_terminal}";
                    // Console.WriteLine(summary);   
                }             
            }
            */

            List<VectorD> history_update_costs = (List<VectorD>)history["update_costs"];

            string output = "";
            foreach (VectorD item in history_update_costs)
            {
                output += $"{item}";                
            }
            // Console.WriteLine(output);

            // test
            // int new_month = QLearning.randomGenerator.Next(0,12);
            // environment.reset(new Hashtable{{"new_month", new_month}});
            // environment.setTypeExecution(TypeExecution.test);
            // qNeuralNetwork.runEpisodes(functionReward, environment);

        }




        public static void tryTest()
        {
            


            deepQLearningDataCenter();


        }
    }

    
}
