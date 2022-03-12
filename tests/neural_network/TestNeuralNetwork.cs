using System;
using System.Collections.Generic;

using layers;
using types;
using custom_lib;
using neural_network;


namespace magodysharp
{
    class TestNeuralNetwork
    {
        
        static void tryTest()
        {

            VectorD v = MathUtil.getVectorRandomNormal(4);
            MatrixD matrix = MathUtil.getMatrixRandomNormal(2,3);


            VectorD v2 = new VectorD{1,2,3};
            MatrixD m2 = new MatrixD();
            m2.Add(v2);
            m2.Add(v2 * 2);
            m2.Add(v2 * 3);

            MatrixD m3 = Weight.getWeights(0,1, new VectorI{3,4}, 10, DenseInitializationMode.xavier);

            
            MatrixD mt1 = new MatrixD();
            mt1.Add(new VectorD{1,2});
            mt1.Add(new VectorD{3,4});
            MatrixD mt2 = new MatrixD();
            mt2.Add(new VectorD{5,7,8});
            mt2.Add(new VectorD{6,9,10});


            
            VectorI vp = new VectorI{4,9,81};

            MatrixD matrix_filled = MatrixD.filled(5,4,(double)0);

            
            MatrixD mt3 = new MatrixD();
            mt3.Add(new VectorD{5,7,8,16});
            mt3.Add(new VectorD{6,9,10,18});
            mt3.Add(new VectorD{1,2,10,19});

            MatrixD mf1 = MatrixD.filled(3,1,(double)6);
            MatrixD mf2 = MatrixD.filled(3,4,(double)6);

            /*
            
            Dictionary<string,string> context = new Dictionary<string, string>();
            MatrixD input = MatrixD.filled(3,5,(double)2);
            Dense dense = new Dense(2, "xavier", input.Count);
            MatrixD output = dense.forward(input,context);
            MatrixD input_gradient = dense.backward(output,0.1f);
            Console.WriteLine(input_gradient);
            
            
            MatrixD input_a = new MatrixD();
            input_a.Add(new VectorD{1,2});
            input_a.Add(new VectorD{-1,-0.2});

            Activation activation1 = new Activation("sigmoid");
            MatrixD output_a = activation1.forward(input_a,context);
            
            MatrixD input_gradient_a = activation1.backward(output_a,0.1f);
            Console.WriteLine(input_gradient);
            */


            // TEST ACTIVATIONS CORRECT CALCULATIONS





            Console.WriteLine(MathUtil.getRandomNormal());

            MatrixD X = new MatrixD();
            X.Add(new VectorD{0,0});
            X.Add(new VectorD{0,1});
            X.Add(new VectorD{1,0});
            X.Add(new VectorD{1,1});
            X = X.transpose();

            
            MatrixD y = new MatrixD();
            y.Add(new VectorD{0});
            y.Add(new VectorD{1});
            y.Add(new VectorD{1});
            y.Add(new VectorD{0});

            int len_data = y.Count;
            int epochs = 1000;
            float learning_rate = 0.01f;
            int batch_size = 2;
            int verbose_level = 10;
            float decay_rate_alpha = 0.1f;

            // 

            Sequential sequentialNetwork = new Sequential(new List<Layer>{
                new Dense(5, DenseInitializationMode.kaiming_he, 2),
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


            /*

            Console.WriteLine(mt1 * mt2 + mt2);
            Console.WriteLine(mt2.transpose());
            Console.WriteLine(mt2.mean(2));
            Console.WriteLine(Vector<int>.sqrt(vp));
            Console.WriteLine(Matrix<double>.sqrt(mt2));
            Console.WriteLine(mt2 + 17);
            Console.WriteLine(Vector<int>.dot(vp,vp*2));
            Console.WriteLine(Matrix<double>.dot(mt1,mt1+2));
            Console.WriteLine(Vector<double>.dotDivide(vp,vp*2));
            Console.WriteLine(Matrix<double>.dotDivide(mt1,mt1));
            Console.WriteLine(mt1 - mt1 * 1.5);
            Console.WriteLine(matrix_filled);
            
                Console.WriteLine(mt3 + mf1); repeat true
                Console.WriteLine(mt3 + mf2); repeat false
            */
        }
    }
}
