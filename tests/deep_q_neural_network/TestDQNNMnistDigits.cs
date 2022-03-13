using System;
using System.Diagnostics;
using System.Collections.Generic;
using System.Collections;

using layers;
using types;
using custom_lib;
using neural_network;
using reinforcement_learning;

using System.IO;

namespace magodysharp
{
    class TestDQNNMnistDigits
    {

        #region DataReaders

        public static DataFormat getDataFromCSV(string path, int samples_max=-1000, bool exist_header=true){

            MatrixD X = new MatrixD();
            MatrixD y = new MatrixD();
            
            using(var reader = new StreamReader(path))
            {
                int line_count = -1;
                while (!reader.EndOfStream)
                {
                    if(line_count == samples_max){
                        break;
                    }
                    line_count += 1;
                    
                    if(line_count == 0){
                        if(exist_header){
                            reader.ReadLine();
                            continue;
                        }
                    }
                    string line = reader.ReadLine();
                    string[] values = line.Split(',');

                    int class_num = Int32.Parse(values[0]);
                    
                    y.Add(Utils.oneHotEncodingVector(class_num,10));

                    VectorD sample = new VectorD();
                    for (int i = 1; i < values.Length; i++)
                    {
                        string value_raw = values[i];
                        if(value_raw.Equals("0")){
                            sample.Add(0.0);
                        }
                        else{
                            double value_pixel_scaled = Double.Parse(value_raw)/255.0;
                            sample.Add(value_pixel_scaled);
                        }
                    }

                    X.Add(sample);
                }
            }

            X = X.transpose();

            DataFormat data = new DataFormat();
            data.X = X;
            data.y = y;
            return data;
        }
            
        #endregion
        

        #region Baselines
        
        public static float accuracyBaselineRandom(MatrixD X_train, MatrixD y_train, MatrixD X_test, MatrixD y_test){
            
            
            float accuracy = 0;
            Random randomGenerator = new Random();
            for (int i = 0; i < y_train.Count; i++)
            {
                int pred = randomGenerator.Next(0,10);
                if((int)y_train[i][pred] == 1){
                    accuracy += 1;
                }
            }
            accuracy /= y_train.Count;
            return accuracy;
        }
        
        #endregion
        

        public static float accuracyQNN(MatrixD X_train, MatrixD y_train, MatrixD X_test, MatrixD y_test){
            
            
            Debug.Assert(X_train.Count == (28*28));
            Debug.Assert(y_train[0].Count == 10);
            Debug.Assert(X_train[0].Count == y_train.Count);

            Debug.Assert(X_test.Count == (28*28));
            Debug.Assert(y_test[0].Count == 10);
            Debug.Assert(X_test[0].Count == y_test.Count);


            float accuracy = 0;
            Random randomGenerator = new Random();

            int num_features = X_train.Count;
            int num_outputs = y_train[0].Count;

            int len_data_train = y_train.Count;
            int len_data_test = y_test.Count;

            int len_data_validation = (int)(len_data_train * 0.2f);
            int len_data_train_new = len_data_train - len_data_validation;
            
            
            // assume is shuffled X_train

            MatrixD X_validation = MatrixD.slice(X_train, 0, num_features-1, len_data_train_new, len_data_train-1);
            MatrixD y_validation = MatrixD.slice(y_train, len_data_train_new, len_data_train-1, 0, num_outputs-1);
            
            X_train = MatrixD.slice(X_train, 0, num_features-1, 0, len_data_train_new-1);
            y_train = MatrixD.slice(y_train, 0, len_data_train_new-1, 0, num_outputs-1);

            X_train = X_train.transpose();
            X_validation = X_validation.transpose();
            ////////////////////MODEL/////////////////
            // HYPERPARAMETERS: vars and functions
           
            int epochs = 3; // epochs are inside neural networks
            float learning_rate = 0.001f;
            int batch_size = 64;
            FunctionLossEnum loss = FunctionLossEnum.mse;
            float gamma = 0.2f;
            float epsilon = 0.2f;
            int total_episodes = 20;
            float target_learning_rate = 0.001f;
            float decay_rate_alpha = (float)(((float)learning_rate/target_learning_rate)-1)/(float)(total_episodes);
            int interval_for_learning = 20; // (int)(EnvironmentDataCenter.timeday_in_minutes/3f);
            int interval_for_update_models = 50; // (int)(EnvironmentDataCenter.timemonth_in_minutes/100f);


            // HYPERPARAMETERS: Topology
            Sequential sequentialNetwork = new Sequential(new List<Layer>{
                new Dense(64,DenseInitializationMode.kaiming_he,num_features),
                new Activation(ActivationFunctionEnum.relu),
                // new Dropout(0.3f),
                new Dense(32, DenseInitializationMode.kaiming_he),
                new Activation(ActivationFunctionEnum.relu),
                // new Dropout(0.2f),
                new Dense(num_outputs)
            });
            
            // Setup
            EnvironmentCommon environmentCommon = new EnvironmentCommon("DQNN_MNIST");
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

            // global
            int index = 0;

            // CALLBACK: Episode: Begin
            Agent.CallbackOnEpisodeBegin onEpisodeBegin = (int episode, EnvironmentRL env) => { };

            Agent.CallbackOnCollectObservations onCollectObservations = (int episode, EnvironmentRL env) => {
                int index_state = (index%len_data_train_new);
                float[] state = X_train[index_state].parseToFloatArray();
                return state;
            };

            Agent.CallbackOnActionPerformed onActionPerformed = (int episode, EnvironmentRL env, float[] action_values) => {
                
                int action = QLearning.selectActionHigher(action_values);
                
                
            };

            QLearning.FunctionReward functionReward = (QLearning self, VectorD state, int action_selected, EnvironmentRL env) => {
                ReplayTuple replayTuple = new ReplayTuple();
                float reward = 0;
                VectorD new_state = new VectorD();

                bool is_terminal = false;

                int index_state = (index%len_data_train_new);

                if(y_train[index_state][action_selected] == 1){
                    reward = 1;
                }
                else{
                    reward = -1;
                }

                // delayed step environment
                index+=1;                
                
                replayTuple.state = state; // old state where action was selected
                replayTuple.action = action_selected;
                replayTuple.reward = reward;
                replayTuple.new_state = state; // already updated in OnActionPerformed
                replayTuple.is_terminal = is_terminal;

                return replayTuple;
            };

            Agent.CallbackOnEpisodeEnd onEpisodeEnd = (int episode, EnvironmentRL env) => {  
                Console.WriteLine($"Episode {episode}: reward {env.reward}. steps {(int)(MathF.Round(env.timestep,3)/0.01f)}. cost {MathF.Round(env.cost,6)}");
            };

            // every step is 0.01 seconds, the episode ends after this counter reach 60
            AgentDQNN agent01 = new AgentDQNN(
                qNeuralNetwork, onEpisodeBegin, onCollectObservations, onActionPerformed,
                onEpisodeEnd, decision_period:0, maximum_steps:len_data_train_new
            );


            agent01.runEpisodesAsync(environmentCommon, qNeuralNetwork.qLearningConfig.total_episodes, functionReward);


            // Prediction

            Console.WriteLine("\nPREDICTION\n");
            MatrixD raw_y_pred = qNeuralNetwork.predict(X_test);
            VectorD res_idx_pred = raw_y_pred.max(1)[1];
            VectorD res_idx_real = y_test.transpose().max(1)[1];

            for (var k = 0; k < res_idx_pred.Count; k++)
            {
                accuracy += (res_idx_pred[k] == res_idx_real[k])? 1:0;
            }
            accuracy /= len_data_test;


            return accuracy;
        }
        
        public static void tryTest()
        {
            Random randomGenerator = new Random();


            string path_file_train = @"data/mnist/handwritten_digits/mnist_train.csv";
            string path_file_test = @"data/mnist/handwritten_digits/mnist_test.csv";

            DataFormat dataTrain = getDataFromCSV(path_file_train, 1000);
            DataFormat dataTest = getDataFromCSV(path_file_test, 100);

            MatrixD X_train = dataTrain.X;
            MatrixD y_train = dataTrain.y;   
            MatrixD X_test = dataTest.X;
            MatrixD y_test = dataTest.y;       

            float accuracy_baseline = TestDQNNMnistDigits.accuracyBaselineRandom(
                X_train,
                y_train,
                X_test,
                y_test
            );

            float accuracy_neural_network = TestDQNNMnistDigits.accuracyQNN(
                X_train,
                y_train,
                X_test,
                y_test
            );

            Console.WriteLine($"Accuracy baseline: {accuracy_baseline*100} %");
            Console.WriteLine($"Accuracy neural network: {accuracy_neural_network*100} %");

        }
    }
}
