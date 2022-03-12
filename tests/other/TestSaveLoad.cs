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
    

    class TestSaveLoad
    {

        
        public static void testSaveLoadJSON(){
            int epochs = 2; // epochs are inside neural networks
            float learning_rate = 0.001f;
            int batch_size = 128;
            FunctionLossEnum loss = FunctionLossEnum.mse;

            float gamma = 0.9f;
            float epsilon = 1;
            int total_episodes = 5;
            int interval_for_learning = (int)(EnvironmentDataCenter.timeday_in_minutes/3f);
            int interval_for_update_models = (int)EnvironmentDataCenter.timemonth_in_minutes;

            int number_actions = 5;

            Sequential sequentialNetwork = new Sequential(new List<Layer>{
                new Dense(64,DenseInitializationMode.kaiming_he,3),
                new Activation(ActivationFunctionEnum.relu),
                new Dropout(0.1f),
                new Dense(32,DenseInitializationMode.kaiming_he),
                new Activation(ActivationFunctionEnum.relu),
                new Dropout(0.1f),
                new Dense(number_actions,DenseInitializationMode.xavier)
            });

            NNConfig nnConfig = new NNConfig(epochs,learning_rate,batch_size,loss);
            QLearningConfig qLearningConfig = new QLearningConfig(
                gamma,epsilon,total_episodes,interval_for_learning, interval_for_update_models
            );

            QNeuralNetwork qNeuralNetwork = new QNeuralNetwork(nnConfig,qLearningConfig,null,sequentialNetwork);
            // qNeuralNetwork.save();
            ReplayTuple replayTuple = new ReplayTuple();
            replayTuple.action = 1;
            replayTuple.is_terminal = false;
            replayTuple.state = new VectorD();
            replayTuple.new_state = new VectorD();
            replayTuple.reward = 3;

            qNeuralNetwork.saveReplayTuple(replayTuple);
            
            
            Hashtable hash = qNeuralNetwork.getJSONHash();
            string file_path = "/home/magody/programming/csharp/LightMLSharp/tempoj.json";
            string json = SerializationJSON.saveJSON(hash,file_path);

            QNeuralNetwork restored = (QNeuralNetwork)QNeuralNetwork.load(file_path);
            Console.WriteLine(restored.ToString());
        }




        public static void tryTest()
        {
            testSaveLoadJSON();


        }
    }

    
}
