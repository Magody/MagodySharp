using System;
using System.Collections.Generic;

using layers;
using types;
using custom_lib;
using neural_network;


namespace magodysharp
{
    class TestNNXOR
    {
        
        public static void tryTest()
        {
            Random randomGenerator = new Random();

            MatrixD X = new MatrixD();
            MatrixD y = new MatrixD();

            for (int i = 0; i < 1000; i++)
            {
                int a = randomGenerator.Next(0,2);
                int b = randomGenerator.Next(0,2);
                X.Add(new VectorD{a,b});
                // Console.WriteLine(a + " , " + b);
                if(a == b){
                    y.Add(new VectorD{0});
                }
                else{
                    y.Add(new VectorD{1});
                }

            }

            X = X.transpose();

            int len_data = y.Count;
            int epochs = 1000;
            float learning_rate = 0.001f;
            int batch_size = 32;
            int verbose_level = 10;
            float decay_rate_alpha = 0.001f;
            
            Sequential sequentialNetwork = new Sequential(new List<Layer>{
                new Dense(4, DenseInitializationMode.kaiming_he, 2),
                new Activation(ActivationFunctionEnum.relu),
                // new Dropout(0.1f),
                new Dense(1),
                new Activation(ActivationFunctionEnum.sigmoid)
            });

            NNConfig nnConfig = new NNConfig(epochs, learning_rate, batch_size, FunctionLossEnum.mse);
            nnConfig.decay_rate_alpha = decay_rate_alpha;

            NeuralNetwork neural_network = new NeuralNetwork(sequentialNetwork, nnConfig);

            Dictionary<string,VectorD> history = neural_network.train(X, y, X, y, verbose_level);

            string key_error = "history_errors";
            Console.WriteLine($"Mean error: {VectorD.mean(history[key_error])}");

            // prediction

            Console.WriteLine("\nPREDICTION\n");
            MatrixD raw_y_pred = neural_network.predict(X);
            VectorD res_idx_pred = raw_y_pred.max(1)[1];
            VectorD res_idx_real = y.transpose().max(1)[1];

            double accuracy = 0;


            
            for (var k = 0; k < res_idx_pred.Count; k++)
            {
                accuracy += (res_idx_pred[k] == res_idx_real[k])? 1:0;
            }
            accuracy /= len_data;


            Console.WriteLine($"Accuracy: {accuracy}\n");

            MatrixD xorGate = new MatrixD();
            xorGate.Add(new VectorD{0,0});
            xorGate.Add(new VectorD{0,1});
            xorGate.Add(new VectorD{1,0});
            xorGate.Add(new VectorD{1,1});
            MatrixD results = neural_network.predict(xorGate.transpose());
            Console.WriteLine(results);

            Console.WriteLine("SUMMARY NET\n" + sequentialNetwork.ToString());


        }
    }
}
