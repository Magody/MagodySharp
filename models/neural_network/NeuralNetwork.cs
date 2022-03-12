using System;
using System.Collections;
using System.Collections.Generic;

using types;
using loss;


using System.Text.Json;


namespace neural_network
{

    public enum FunctionLossEnum
    {
        mse, softmax_cross_entropy
    }
    public class NNConfig
    {
        public int epochs {get; set;}
        public float learning_rate {get; set;}
        public int batch_size {get; set;}
        public int lambda {get; set;}

        public float decay_rate_alpha {get; set;}


        private string loss_name {get;set;}

        public delegate double LossCost(MatrixD y_true, MatrixD y_pred);
        public delegate MatrixD LossGradient(MatrixD y_true, MatrixD y_pred);

        public LossCost functionLossCost {get; set;}
        public LossGradient functionLossGradient {get; set;}
        public NNConfig(){
            
        }
        public NNConfig(
            int epochs, float learning_rate, int batch_size, FunctionLossEnum loss,
            int lambda=0, float decay_rate_alpha=0.1f
        ){
            this.epochs = epochs;
            this.learning_rate = learning_rate;
            this.batch_size = batch_size;
            this.lambda = lambda;
            this.decay_rate_alpha = decay_rate_alpha;

            this.loss_name = $"{loss}";

            switch (loss)
            {
                case FunctionLossEnum.mse:
                    this.functionLossCost = Loss.mse;
                    this.functionLossGradient = Loss.mseDerivative;
                    break;
                case FunctionLossEnum.softmax_cross_entropy:
                    this.functionLossCost = Loss.binaryCrossEntropy;
                    this.functionLossGradient = Loss.softmaxGradient;
                    break;
                default:
                    throw new Exception("not supported loss function");
            }

        }

        /*
        Pending to adequate

        % ADAM optimizer
        b1 = 0.9;
        b2 = 0.999;
        */

        public Hashtable getJSONHash(){
            Hashtable hash = new Hashtable();
            hash["epochs"] = this.epochs;
            hash["learning_rate"] = this.learning_rate;
            hash["batch_size"] = this.batch_size;
            hash["lambda"] = this.lambda;
            hash["decay_rate_alpha"] = this.decay_rate_alpha;
            hash["loss_name"] = $"{this.loss_name}";

            return hash;
        }

        public static NNConfig buildNNConfig(JsonElement element){
            
            string loss = element.GetProperty("loss_name").GetString();


            FunctionLossEnum functionLossEnum = FunctionLossEnum.mse;

            switch (loss)
            {
                case "mse":
                    functionLossEnum = FunctionLossEnum.mse;
                    break;
                case "softmax_cross_entropy":
                    functionLossEnum = FunctionLossEnum.softmax_cross_entropy;
                    break;
            }


            NNConfig nnConfig = new NNConfig(
                element.GetProperty("epochs").GetInt32(),
                (float)element.GetProperty("learning_rate").GetDouble(),
                element.GetProperty("batch_size").GetInt32(),
                functionLossEnum,
                element.GetProperty("lambda").GetInt32(),
                (float)element.GetProperty("decay_rate_alpha").GetDouble()

            );


            return nnConfig;

        }

    }
    
    public class NeuralNetwork
    {

        public Sequential sequentialNetwork {get; set;}
        public NNConfig nnConfig {get; set;}

        public float alpha {get; set;}

        public NeuralNetwork(Sequential sequentialNetwork, NNConfig nnConfig){
            this.sequentialNetwork = sequentialNetwork;
            this.nnConfig = nnConfig;
            this.alpha = this.nnConfig.learning_rate;
        }

        public Dictionary<string,VectorD> train(MatrixD X_train, MatrixD y_train, MatrixD X_validation, MatrixD y_validation, int verbose_level=1){
            // For 2 dimensions only

            Hashtable context = new Hashtable{
                 {"is_test", "false"}
            };
            
            Dictionary<string,VectorD> history = new Dictionary<string,VectorD>();


            VectorD history_errors = new VectorD();
            VectorD history_accuracy_validation = new VectorD();
            
            VectorI shape_input = new VectorI{X_train.Count, X_train[0].Count};
            
            int len_data_train = shape_input[1];
            int num_batchs = (int)Math.Ceiling((double)len_data_train/this.nnConfig.batch_size);

            for (var e = 0; e < this.nnConfig.epochs; e++)
            {
                double error = 0;
                for (int index_data = 0; index_data < len_data_train; index_data+=this.nnConfig.batch_size)
                {

                    int batch_end = index_data+this.nnConfig.batch_size-1;


                    VectorI batch_range = new VectorI{index_data,batch_end};
                    if(batch_end > len_data_train)
                        batch_range[1] = len_data_train-1;

                    
                    MatrixD x = MatrixD.slice(X_train,0,X_train.Count-1,batch_range[0],batch_range[1]);

                    
                    MatrixD y = MatrixD.slice(y_train,batch_range[0],batch_range[1],0,y_train[0].Count-1);
                    
                    // each col is a prediction
                    MatrixD output = this.forwardFull(x, context);

                    MatrixD yt = y.transpose();
                    
                    // error
                    error += this.nnConfig.functionLossCost(yt, output)/this.nnConfig.batch_size;
                    
                    this.backward(yt, output);
                    

                }
                
                double accuracy = 0;

                if(X_validation.Count > 0){
                    
                    MatrixD raw_y_validation = this.predict(X_validation);

                    VectorD res_idx_pred = raw_y_validation.max(1)[1];
                    VectorD res_idx_real = y_validation.transpose().max(1)[1];

                    
                    for (var k = 0; k < res_idx_pred.Count; k++)
                    {
                        accuracy += (res_idx_pred[k] == res_idx_real[k])? 1:0;
                    }
                    accuracy /= res_idx_pred.Count;
                    history_accuracy_validation.Add(accuracy);
                }               
                this.alpha = this.nnConfig.learning_rate/(1 + this.nnConfig.decay_rate_alpha * e);

                error = error / num_batchs;

                
                
                if(double.IsNaN(error)){
                    throw new Exception("Gradient exploding!");
                }                

                history_errors.Add(error);

                if(e % (int)Math.Floor((float)this.nnConfig.epochs/10) == 0 && verbose_level > 0){
                    Console.WriteLine($"{e}/{this.nnConfig.epochs}, error={error}, val_acc={accuracy}");
                }
            }

            history["history_errors"] = history_errors;
            history["history_accuracy_validation"] = history_accuracy_validation;



            return history;


        }

        
        // this methods will be overriden by childs
        public virtual MatrixD forwardFull(MatrixD x, Hashtable context){
            return this.sequentialNetwork.forward(x,context);
        }

        public virtual MatrixD backward(MatrixD y, MatrixD output){

            int len_network = this.sequentialNetwork.network.Count;

            MatrixD grad = this.nnConfig.functionLossGradient(y, output);
            for (var index_layer = len_network-1; index_layer >= 0; index_layer--)
            {
                grad = this.sequentialNetwork.network[index_layer].backward(grad, this.alpha);
            }

            return grad;

        }


        public virtual MatrixD predict(MatrixD X){
            Hashtable context = new Hashtable{
                 {"is_test", "true"}
            };
            return this.forwardFull(X, context);
        }
        

    }

}