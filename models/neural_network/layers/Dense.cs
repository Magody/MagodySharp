using System;

using System.Collections;
using System.Collections.Generic;


using types;
using custom_lib;


using System.Text.Json;


namespace layers
{
    
    public class Dense:Layer
    {
        #region 
        // PROPERTIES
        // - Inherit
        public override VectorI shape_input {get; set;}
        public override VectorI shape_output {get; set;}

        // - Own
        public DenseInitializationMode mode {get; set;}
        public MatrixD input {get; set;}
        public MatrixD weights {get; set;}
        public MatrixD bias {get; set;}

        private float init_mean = 0f;
        private float init_std = 1f;

        // -- ADAM optimizer
        public MatrixD vdw {get; set;}
        public MatrixD vdb {get; set;}
        public MatrixD sdw {get; set;}
        public MatrixD sdb {get; set;}

        #endregion
        

        public Dense(int neurons_output, DenseInitializationMode mode=DenseInitializationMode.xavier, int neurons_input=-1){
            
            this.shape_output = new VectorI{neurons_output, 1};
            this.mode = mode;

            if(neurons_input != -1){
                this.init(new VectorI{neurons_input, 1});
            }

        }

        public override VectorI init(VectorI shape_input){
            this.shape_input = shape_input;
            
            this.weights = Weight.getWeights(this.init_mean, this.init_std, new VectorI{this.shape_output[0], this.shape_input[0]}, this.shape_input[0], this.mode);
            // this.bias = zeros([shape_output(1), 1]);
            this.bias = Weight.getWeights(this.init_mean, this.init_std, new VectorI{this.shape_output[0], 1}, this.shape_input[0], this.mode);
        
            int nw = this.weights.Count, mw = this.weights[0].Count;
            int nb = this.bias.Count, mb = this.bias[0].Count;

            this.vdw = MatrixD.filled(nw,mw,0);
            this.vdb = MatrixD.filled(nb,mb,0);
            this.sdw = MatrixD.filled(nw,mw,0);
            this.sdb = MatrixD.filled(nb,mb,0);


            return this.shape_output;

        }

        public override Layer clone(){
            Dense denseClone = new Dense(this.shape_output[0],this.mode);
            denseClone.shape_input = VectorI.clone(this.shape_input);
            denseClone.input = MatrixD.clone(this.input);
            denseClone.weights = MatrixD.clone(this.weights);
            denseClone.bias = MatrixD.clone(this.bias);
            denseClone.init_mean = this.init_mean;
            denseClone.init_std = this.init_std;
            denseClone.vdw = MatrixD.clone(this.vdw);
            denseClone.vdb = MatrixD.clone(this.vdb);
            denseClone.sdw = MatrixD.clone(this.sdw);
            denseClone.sdb = MatrixD.clone(this.sdb);

            return denseClone;
        }

        public override MatrixD forward(MatrixD input, Hashtable context){
            // input = X => each col is an example
            this.input = input;
            
            MatrixD output = this.weights * this.input + this.bias;

            return output;
        }
        public override MatrixD backward(MatrixD output_gradient, float learning_rate){
        
            MatrixD weights_gradient = output_gradient * this.input.transpose();
            MatrixD bias_gradient = output_gradient.mean(2);


            MatrixD input_gradient = this.weights.transpose() * output_gradient;

            // Adam
            double b1 = 0.9;
            double b2 = 0.999;
            double eps = 1e-8;

            // update momentum
            this.vdw = (this.vdw * b1) + (weights_gradient * (1-b1));
            this.vdb = (this.vdb * b1) + (bias_gradient * (1-b1));
            // update RMSprop
            // weights_gradient.multiply(weights_gradient) = weights_gradient .^ 2, square each element
            this.sdw = (this.sdw * b2) + (MatrixD.dot(weights_gradient, weights_gradient) * (1-b2));
            this.sdb = (this.sdb * b2) + (MatrixD.dot(bias_gradient, bias_gradient) * (1-b2));

            /*
            % bias correction
            this.vdw = this.vdw ./ (1 - (b1 ^ t));
            this.vdb = this.vdb ./ (1 - (b1 ^ t));
            this.sdw = this.sdw ./ (1 - (b2 ^ t));
            this.sdb = this.sdb ./ (1 - (b2 ^ t));
            */

            this.weights = this.weights - (MatrixD.dotDivide(this.vdw, MatrixD.sqrt(this.sdw) + eps) * learning_rate);
            this.bias = this.bias - (MatrixD.dotDivide(this.vdb,MatrixD.sqrt(this.sdb) + eps) * learning_rate);
            

            /*
            self.weights = self.weights - learning_rate * weights_gradient;
            self.bias = self.bias - learning_rate * bias_gradient;
            
            self.weights = self.weights - learning_rate * (self.vdw./(sqrt(self.sdw) + eps));
            self.bias = self.bias - learning_rate * (self.vdb./(sqrt(self.sdb) + eps));
            */
            return input_gradient;
        }

        public override Hashtable getJSONHash()
        {

            Hashtable hash = new Hashtable();
            hash["layer_type"] = "dense";
            hash["shape_input"] = this.shape_input;
            hash["shape_output"] = this.shape_output;
            hash["mode"] = $"{this.mode}";
            hash["input"] = (this.input == null)? new MatrixD():this.input;
            hash["weights"] = (this.weights == null)? new MatrixD():this.weights;
            hash["bias"] = (this.bias == null)? new MatrixD():this.bias;
            hash["init_mean"] = this.init_mean;
            hash["init_std"] = this.init_std;
            hash["vdw"] = (this.vdw == null)? new MatrixD():this.vdw;
            hash["vdb"] = (this.vdb == null)? new MatrixD():this.vdb;
            hash["sdw"] = (this.sdw == null)? new MatrixD():this.sdw;
            hash["sdb"] = (this.sdb == null)? new MatrixD():this.sdb;

            return hash;
        }

        public static Layer buildLayer(JsonElement element){

            string mode_string = element.GetProperty("mode").GetString();
            DenseInitializationMode denseInitializationMode = DenseInitializationMode.xavier;

            switch (mode_string)
            {
                case "xavier":
                    denseInitializationMode = DenseInitializationMode.xavier;
                    break;
                case "kaiming_he":
                    denseInitializationMode = DenseInitializationMode.kaiming_he;
                    break;
            }

            
            Dense layer = new Dense(1,denseInitializationMode);
            layer.shape_input = VectorI.parseJsonElement(element.GetProperty("shape_input"));
            layer.shape_output = VectorI.parseJsonElement(element.GetProperty("shape_output"));
            layer.input = MatrixD.parseJsonElement(element.GetProperty("input"));
            layer.weights = MatrixD.parseJsonElement(element.GetProperty("weights"));
            layer.bias = MatrixD.parseJsonElement(element.GetProperty("bias"));
            layer.init_mean = (float)element.GetProperty("init_mean").GetDouble();
            layer.init_std = (float)element.GetProperty("init_std").GetDouble();
            layer.vdw = MatrixD.parseJsonElement(element.GetProperty("vdw"));
            layer.vdb = MatrixD.parseJsonElement(element.GetProperty("vdb"));
            layer.sdw = MatrixD.parseJsonElement(element.GetProperty("sdw"));
            layer.sdb = MatrixD.parseJsonElement(element.GetProperty("sdb"));


            return layer;

        }

        public override string ToString(){

            string output = $"\nshape_input={this.shape_input},";
            output += $"\nshape_output={this.shape_output}";
            // output += $"\nweights={this.weights}";
            // output += $"\nbias={this.bias}";

            return output;


        }

    
    }
}

