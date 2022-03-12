using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;

using types;
using custom_lib;

using neural_network;

using System.Text.Json;


namespace reinforcement_learning
{

    
    public struct ExperienceReplayHistoryLearning
    {  
        public double mean_cost;
        public bool learned;                
    }
    public class QNeuralNetwork:QLearning
    {
        // Neural network parameters
        public VectorI shape_input {get; set;}
        public VectorI shape_output {get; set;}

        public Sequential sequential_conv_network {get; set;} // here is an optional network, null for ignore it
        public Sequential sequential_network {get; set;}
        
        public Sequential sequential_conv_network_target {get; set;}
        public Sequential sequential_network_target {get; set;}
        
        public NNConfig nnConfig {get; set;}
        
        // alpha decay
        public float alpha {get; set;}
        
        // aux
        public bool use_convolutional {get; set;}
        public int actions_length {get; set;}


        public QNeuralNetwork(
            NNConfig nnConfig, 
            QLearningConfig qLearningConfig,
            QLearning.FunctionEpisode functionEpisode,
            Sequential sequential_conv_network,
            Sequential sequential_network
        ):base(qLearningConfig, functionEpisode){
            // Config parameters
            // NN and QNN config
            this.nnConfig = nnConfig;
            // decay epsilon/alpha
            this.epsilon = this.qLearningConfig.initial_epsilon;
            this.alpha = this.nnConfig.learning_rate;

            this.actions_length = sequential_network.shape_output[0];  // [actions, 1]
            
            this.initGameReplay(this.actions_length);
            this.initNetwork(sequential_conv_network, sequential_network);
        }

        public QNeuralNetwork(
            NNConfig nnConfig, 
            QLearningConfig qLearningConfig,
            Sequential sequential_conv_network = null,
            Sequential sequential_network = null
        ):base(qLearningConfig){
            // Config parameters
            // NN and QNN config
            this.nnConfig = nnConfig;
            // decay epsilon/alpha
            this.epsilon = this.qLearningConfig.initial_epsilon;
            this.alpha = this.nnConfig.learning_rate;

            this.actions_length = sequential_network.shape_output[0];  // [actions, 1]
            
            this.initGameReplay(this.actions_length);
            this.initNetwork(sequential_conv_network, sequential_network);
        }

        public void initNetwork(
            Sequential sequential_conv_network,
            Sequential sequential_network
        ){
            // used for begining or transfer learning
            this.use_convolutional = sequential_conv_network != null;
        
            if(this.use_convolutional)
                this.shape_input = sequential_conv_network.network[0].shape_input;
            else
                this.shape_input = sequential_network.network[0].shape_input;
                
            this.shape_output = sequential_network.network[sequential_network.network.Count-1].shape_output;
            
            int prod = 1;
            for (var i = 0; i < this.shape_output.Count; i++)
            {
                prod *= this.shape_output[i];
            }

            this.actions_length = prod;
            
            this.sequential_conv_network = sequential_conv_network;
            this.sequential_network = sequential_network;
            
            // theta freeze
            if(use_convolutional){
                this.sequential_conv_network_target = sequential_conv_network.clone(); 
            }
            else{
                this.sequential_conv_network_target = null; 
            }
            
            this.sequential_network_target = sequential_network.clone(); 
        }



        public void updateQNeuralNetworkTarget(){
            if(use_convolutional){
                this.sequential_conv_network_target = this.sequential_conv_network.clone(); 
            }
            else{
                this.sequential_conv_network_target = null; 
            }
            Sequential sequentialClone = sequential_network.clone(); 
            this.sequential_network_target = sequentialClone;

        }

        public Hashtable train(MatrixD X, MatrixD Y, int episode, int verbose_level=1){
            Hashtable context = new Hashtable{
                {"is_test",false}
            }; 
            
            // is trained as multiclass classification
            
            
            Hashtable history = new Hashtable();
            
            VectorD history_errors = new VectorD();
            
            int len_data_train = X[0].Count;
            
            
            int num_batchs = (int)Math.Ceiling((double)len_data_train/this.nnConfig.batch_size);

            for (var epoch = 0; epoch < this.nnConfig.epochs; epoch++)
            {
                double error = 0;
                for (int index_data = 0; index_data < len_data_train; index_data+=this.nnConfig.batch_size)
                {

                    int batch_end = index_data+this.nnConfig.batch_size-1;

                    VectorI batch_range = new VectorI{index_data,batch_end};
                    if(batch_end > len_data_train)
                        batch_range[1] = len_data_train-1;

                    
                    MatrixD x = MatrixD.slice(X,0,X.Count-1,batch_range[0],batch_range[1]);

                    MatrixD y = MatrixD.slice(Y,batch_range[0],batch_range[1],0,Y[0].Count-1);
                    
                    // each col is a prediction
                    MatrixD output = this.forwardFull(x, context);

                    MatrixD yt = y.transpose();
                    
                    // error
                    double error_complete = this.nnConfig.functionLossCost(yt, output);
                    error +=(error_complete/this.nnConfig.batch_size);
                    
                    
                    MatrixD grad = this.nnConfig.functionLossGradient(yt, output);
                    grad = this.sequential_network.backward(grad, this.alpha);

                    if(this.use_convolutional){
                        this.sequential_conv_network.backward(grad, this.alpha);
                    }

                }

                
                error = error / num_batchs;
                
                history_errors.Add(error);
            }
            
            // epsilon decay
            this.updateEpsilonDecay(episode);

            // alpha decay
            this.alpha = this.nnConfig.learning_rate/(1 + this.nnConfig.decay_rate_alpha * episode);
                

            history["history_errors"] = history_errors;

            return history;
        }

        public MatrixD forwardFull(MatrixD x, Hashtable context){
            MatrixD features = x;
            
            if(this.use_convolutional){
                features = this.sequential_conv_network.forward(x, context);
            }
            return this.sequential_network.forward(features, context);                     
        }

        public MatrixD predict(MatrixD X){
            Hashtable context = new Hashtable{
                {"is_test",true}
            };
            return this.forwardFull(X, context);
        }
        
        public QSelection selectAction(VectorD state, bool is_test){
            Hashtable context = new Hashtable{
                {"is_test",is_test}
            };

            MatrixD x_single = new MatrixD();
            x_single.Add(state);
            x_single = x_single.transpose(); // todo: this transformations...

            MatrixD Qval = this.forwardFull(x_single, context).transpose();
            return QLearning.selectActionQEpsilonGreedy(Qval[0], this.epsilon, this.actions_length, is_test);
            
        }

        public ExperienceReplayHistoryLearning learnFromExperienceReplay(int episode, int verbose_level=1){



            Hashtable context = new Hashtable{
                {"is_test",false}
            };

            ExperienceReplayHistoryLearning history_learning = new ExperienceReplayHistoryLearning();
           history_learning.learned = false;
            
            if(gameReplayNumValidElements < this.nnConfig.batch_size)
                return history_learning;

            // no reference
            ReplayTuple[] gameReplayBatch = this.getGameReplayBatch(this.nnConfig.batch_size);

                        
            int input_dataX = this.shape_input[0];
            
            // dataX should be transposed
            MatrixD dataX = MatrixD.zeros(this.nnConfig.batch_size, input_dataX);
            MatrixD dataY = MatrixD.zeros(this.nnConfig.batch_size, this.actions_length);
            
            // Computations for the minibatch
            for (var numExample = 0; numExample < this.nnConfig.batch_size; numExample++)
            {
                ReplayTuple sample = gameReplayBatch[numExample];
                // Getting the value of Q(s, a)
                VectorD s = sample.state;
                VectorD s_Qval = this.forwardFull(MatrixD.transposeVector(s), context).transpose()[0];
                
                // Getting the value of max_a_Q(s',a"]
                VectorD s_prime = sample.new_state;

                MatrixD inp = MatrixD.transposeVector(s_prime);
                if(this.use_convolutional){
                    inp = this.sequential_conv_network_target.forward(inp, context);
                }
                
                VectorD s_prime_Qval = this.sequential_network_target.forward(inp, context).transpose()[0];  // sequential_network_target.forward
                double maxQval_er = VectorD.max(s_prime_Qval)["value"];
                
                // selected action and reward
                int action_er = sample.action;
                double reward_er = sample.reward;
                double update_er = 0;

                bool is_terminal = sample.is_terminal;
                if(is_terminal){
                    // Terminal state
                    update_er = reward_er;
                    
                }                    
                else{
                    // Non-terminal state
                    update_er = reward_er + this.qLearningConfig.gamma*maxQval_er;
                }

                dataX[numExample] = s;                
                dataY[numExample] = s_Qval;
                dataY[numExample][action_er] = update_er;
                
                /*
                if(is_terminal && action_er == 2){
                    Console.WriteLine($"Bad updated {Math.Round(s[0],2)}: {dataY[numExample]}");
                }
                */
            }

            dataX = dataX.transpose();
            
            Hashtable history = this.train(dataX, dataY, episode, verbose_level-1);
            
            
            VectorD history_errors = (VectorD)history["history_errors"];
            history_learning.mean_cost = VectorD.mean(history_errors);
            history_learning.learned = true;

            return history_learning;
        }

        public override Hashtable getJSONHash(){
           

            Hashtable hash = new Hashtable();
            hash["shape_input"] = this.shape_input;
            hash["shape_output"] = this.shape_output;

            // QLearning inherit
            hash["useCustomRunEpisodes"] = this.useCustomRunEpisodes;
            hash["qLearningConfig"] = this.qLearningConfig.getJSONHash();


            hash["gameReplay"] = this.gameReplay;
            hash["gameReplayUsage"] = this.gameReplayUsage;
            hash["gameReplayCounter"] = this.gameReplayCounter;
            hash["gameReplayNumValidElements"] = this.gameReplayNumValidElements;

            hash["index_action"] = this.index_action;
            hash["epsilon"] = this.epsilon;


            // qnn specific parameters
            hash["sequential_conv_network"] = (this.use_convolutional)? this.sequential_conv_network.getJSONHash():new Hashtable();
            hash["sequential_network"] = this.sequential_network.getJSONHash();

            hash["sequential_conv_network_target"] = (this.use_convolutional)? this.sequential_conv_network_target.getJSONHash():new Hashtable();
            hash["sequential_network_target"] = this.sequential_network_target.getJSONHash();

            hash["nnConfig"] = this.nnConfig.getJSONHash();

            hash["alpha"] = this.alpha;

            hash["use_convolutional"] = this.use_convolutional;
            hash["actions_length"] = this.actions_length;

            return hash;
        }

        public override string save(string path_file_name="qnn.json", bool append = false){
            Hashtable hash = this.getJSONHash();
            string json = SerializationJSON.saveJSON(hash,path_file_name);
            return json;
        }

        public new static QLearning load(string path_file_name="qnn.json"){
            
            JsonElement jsonRoot = SerializationJSON.loadJSON(path_file_name);

            NNConfig nNConfig = NNConfig.buildNNConfig(jsonRoot.GetProperty("nnConfig"));
            QLearningConfig qLearningConfig = QLearningConfig.buildQLearningConfig(jsonRoot.GetProperty("qLearningConfig"));
            Sequential sequentialConv = Sequential.buildSequential(jsonRoot.GetProperty("sequential_conv_network"));
            Sequential sequential = Sequential.buildSequential(jsonRoot.GetProperty("sequential_network"));


            QNeuralNetwork qnn = new QNeuralNetwork(
                nNConfig, qLearningConfig, null, sequentialConv, sequential
            );
            qnn.shape_input = VectorI.parseJsonElement(jsonRoot.GetProperty("shape_input"));
            qnn.shape_output = VectorI.parseJsonElement(jsonRoot.GetProperty("shape_output"));

            qnn.useCustomRunEpisodes = jsonRoot.GetProperty("useCustomRunEpisodes").GetBoolean();
            qnn.gameReplay = JsonSerializer.Deserialize<ReplayTuple?[]>(jsonRoot.GetProperty("gameReplay").GetRawText());

            qnn.gameReplayUsage = JsonSerializer.Deserialize<bool[]>(jsonRoot.GetProperty("gameReplayUsage").GetRawText());

            qnn.gameReplayCounter = jsonRoot.GetProperty("gameReplayCounter").GetInt32();
            qnn.gameReplayNumValidElements = jsonRoot.GetProperty("gameReplayNumValidElements").GetInt32();


            qnn.index_action = JsonSerializer.Deserialize<int[]>(jsonRoot.GetProperty("index_action").GetRawText());
            qnn.epsilon = (float)jsonRoot.GetProperty("epsilon").GetDouble();

            qnn.sequential_conv_network_target =  Sequential.buildSequential(jsonRoot.GetProperty("sequential_conv_network_target"));
            qnn.sequential_network_target = Sequential.buildSequential(jsonRoot.GetProperty("sequential_network_target"));


            qnn.alpha = (float)jsonRoot.GetProperty("alpha").GetDouble();

            qnn.use_convolutional = jsonRoot.GetProperty("use_convolutional").GetBoolean();

            qnn.actions_length = jsonRoot.GetProperty("actions_length").GetInt32();

            return qnn;
        }


        public override string ToString(){

            string output = "QNeuralNetwork: ";

            ReplayTuple? replayTuple = this.gameReplay[0];

            if(replayTuple != null){
                output += $"{replayTuple?.reward}";
            }

            return output;
        }
        
    }
}
