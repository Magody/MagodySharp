using System;

using System.Collections;
using System.Collections.Generic;

using types;
using custom_lib;


using System.Text.Json;


namespace layers
{

    public enum ActivationFunctionEnum
    {
        sigmoid, tanh, relu, elu, softmax, none
    }


    public class Activation:Layer
    {
        #region 
        // PROPERTIES
        // - Inherit
        public override VectorI shape_input {get; set;}
        public override VectorI shape_output {get; set;}

        // - Own
        public MatrixD input {get; set;}
        public delegate MatrixD ActivationFunction(MatrixD x);
        public delegate MatrixD ActivationFunctionDerivative(MatrixD x);

        public ActivationFunction activation {get; set;}
        public ActivationFunctionDerivative activationDerivative {get; set;}

        private ActivationFunctionEnum activation_name;

        #endregion
        

        public Activation(ActivationFunctionEnum activation_name){
            this.activation_name = activation_name;
            switch (this.activation_name)
            {
                case ActivationFunctionEnum.sigmoid:
                    this.activation = Activation.sigmoid;
                    this.activationDerivative = Activation.sigmoidDerivative;
                    break;
                case ActivationFunctionEnum.tanh:
                    this.activation = Activation.tanh;
                    this.activationDerivative = Activation.tanhDerivative;
                    break;
                case ActivationFunctionEnum.relu:
                    this.activation = Activation.relu;
                    this.activationDerivative = Activation.reluDerivative;
                    break;
                case ActivationFunctionEnum.elu:
                    this.activation = Activation.elu;
                    this.activationDerivative = Activation.eluDerivative;
                    break;
                case ActivationFunctionEnum.softmax:
                    this.activation = Activation.softmax;
                    this.activationDerivative = Activation.softmaxDerivative;
                    break;
                default:
                    this.activation = Activation.relu;
                    this.activationDerivative = Activation.reluDerivative;
                    break;
            }

        }

        public override VectorI init(VectorI shape_input){
            this.shape_input = shape_input;
            this.shape_output = VectorI.clone(shape_input);
            return this.shape_output;

        }

        public override Layer clone(){
            Activation activationClone = new Activation(this.activation_name);
            activationClone.init(VectorI.clone(this.shape_input));
            return activationClone;
        }

        public override MatrixD forward(MatrixD input, Hashtable context){
            
            this.input = input;
            MatrixD output = this.activation(input);

            return output;
        }
        public override MatrixD backward(MatrixD output_gradient, float learning_rate){
        
            if(activation_name == ActivationFunctionEnum.softmax){
                return output_gradient;
            }
            
            return MatrixD.dot(output_gradient, this.activationDerivative(input));

            
        }

         public override string ToString(){

            string output = $"Activation: {this.activation_name}";


            return output;
        }


        public static MatrixD sigmoid(MatrixD x){
            int n = x.Count, m = x[0].Count;

            MatrixD y = MatrixD.filled(n,m,(double)1);

            y = MatrixD.dotDivide(y, (MatrixD.exp(x*-1) + 1));
            return y;
        }

        public static MatrixD sigmoidDerivative(MatrixD x){
            int n = x.Count, m = x[0].Count;

            MatrixD s = Activation.sigmoid(x);

            MatrixD m_ones = MatrixD.filled(n,m,(double)1);
            MatrixD y = MatrixD.dot(s, (m_ones - s));
            return y;
        }

        public static MatrixD tanh(MatrixD x){
            int n = x.Count, m = x[0].Count;

            
            MatrixD y = MatrixD.tanh(x);

            return y;
        }

        public static MatrixD tanhDerivative(MatrixD x){
            int n = x.Count, m = x[0].Count;

            MatrixD y = MatrixD.ones(n,m) - MatrixD.dot(MatrixD.tanh(x), MatrixD.tanh(x));

            return y;
        }


        public static MatrixD relu(MatrixD x){
            int n = x.Count, m = x[0].Count;

            MatrixD y = MatrixD.zeros(n,m);
            for (var i = 0; i < n; i++)
            {
                for (var j = 0; j < m; j++)
                {
                    y[i][j] = Math.Max(0,x[i][j]);
                }
                
            }

            return y;
        }
        
        public static MatrixD reluDerivative(MatrixD x){
            int n = x.Count, m = x[0].Count;

            MatrixD y = MatrixD.zeros(n,m);
            for (var i = 0; i < n; i++)
            {
                for (var j = 0; j < m; j++)
                {
                    if(x[i][j] > 0){
                        y[i][j] = 1;
                    }
                }
                
            }

            return y;
        }

        
        public static MatrixD softmax(MatrixD x){
            int n = x.Count, m = x[0].Count;


            MatrixD ex = MatrixD.exp(x);
            MatrixD ex_sum = ex.sum(1);


            MatrixD y = MatrixD.dotDivide(ex, ex_sum);
            return y;
        }

        public static MatrixD softmaxDerivative(MatrixD x){
            int n = x.Count, m = x[0].Count;

            // the cross entropy is made outside with Class Loss
            MatrixD y = x;
            return y;
        }

        public static MatrixD elu(MatrixD x){
            int n = x.Count, m = x[0].Count;

            MatrixD y = MatrixD.zeros(n,m);
            for (var i = 0; i < n; i++)
            {
                for (var j = 0; j < m; j++)
                {
                    double val = x[i][j];
                    y[i][j] = Math.Max(0,val) * val + ((val<=0)?1:0) * (Math.Exp(val) - 1);
                }
                
            }

            return y;
        }
        
        public static MatrixD eluDerivative(MatrixD x){
            int n = x.Count, m = x[0].Count;
            float threshold = 0;

            MatrixD y = MatrixD.zeros(n,m);
            for (var i = 0; i < n; i++)
            {
                for (var j = 0; j < m; j++)
                {
                    if(x[i][j] > threshold){
                        y[i][j] = 1;
                    }
                }
                
            }

            return y;
        }

        public override Hashtable getJSONHash()
        {

            Hashtable hash = new Hashtable();
            hash["layer_type"] = "activation";
            hash["shape_input"] = this.shape_input;
            hash["shape_output"] = this.shape_output;
            hash["input"] = (this.input == null)? new MatrixD():this.input;
            hash["activation_name"] = $"{activation_name}";

            /*

            public ActivationFunction activation {get; set;}
            public ActivationFunctionDerivative activationDerivative {get; set;}

            */
            return hash;
        }


        public static Layer buildLayer(JsonElement element){

            string activation_name_string = element.GetProperty("activation_name").GetString();
            ActivationFunctionEnum activationFunctionEnum = ActivationFunctionEnum.relu;

            switch (activation_name_string)
            {
                case "sigmoid":
                    activationFunctionEnum = ActivationFunctionEnum.sigmoid;
                    break;
                case "tanh":
                    activationFunctionEnum = ActivationFunctionEnum.tanh;
                    break;
                case "relu":
                    activationFunctionEnum = ActivationFunctionEnum.relu;
                    break;
                case "elu":
                    activationFunctionEnum = ActivationFunctionEnum.elu;
                    break;
                case "softmax":
                    activationFunctionEnum = ActivationFunctionEnum.softmax;
                    break;
            }

            
            Activation layer = new Activation(activationFunctionEnum);
            layer.shape_input = VectorI.parseJsonElement(element.GetProperty("shape_input"));
            layer.shape_output = VectorI.parseJsonElement(element.GetProperty("shape_output"));
            layer.input = MatrixD.parseJsonElement(element.GetProperty("input"));

            return layer;

        }
    
    }
}

