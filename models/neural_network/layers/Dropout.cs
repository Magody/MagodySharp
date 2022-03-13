using System;

using System.Collections;
using System.Collections.Generic;

using types;
using custom_lib;

using System.Text.Json;

namespace layers
{
    public class Dropout:Layer
    {
        #region 
        // PROPERTIES
        // - Inherit
        public override VectorI shape_input {get; set;}
        public override VectorI shape_output {get; set;}

        // - Own

        public MatrixD mask {get; set;}
        public float dropout_rate {get; set;}

        #endregion

        private Random randomGenerator;
        
        public Dropout(float dropout_rate){
            this.dropout_rate = dropout_rate;
            this.randomGenerator = new Random();
        }
        public Dropout(float dropout_rate, VectorI shape_input){
            this.dropout_rate = dropout_rate;

            if(shape_input.Count > 0){
                this.init(shape_input);
            }
        }

        public override VectorI init(VectorI shape_input){
            this.shape_input = shape_input;
            this.shape_output = shape_input;
            return this.shape_output;

        }

        public override Layer clone(){
            Dropout dropoutClone = new Dropout(this.dropout_rate);
            dropoutClone.init(VectorI.clone(this.shape_input));
            dropoutClone.mask = MatrixD.clone(this.mask);
            return dropoutClone;
        }

        public override MatrixD forward(MatrixD input, Hashtable context){
            
            int m = input[0].Count;

            MatrixD output;

            bool is_test = Boolean.Parse(context["is_test"].ToString());
            
            if(is_test)
                output = input;
            else{
                float keep_probability = 1 - this.dropout_rate;
                // get random from uniform, vectorized operation
                // TODO: change seed
                // this.mask = binornd(1, keep_probability * ones(this.shape_input));

                this.mask = new MatrixD();
                for (var i = 0; i < this.shape_input[0]; i++)
                {
                    VectorD v = new VectorD();
                    for (var j = 0; j < this.shape_input[1]; j++)
                    {
                        v.Add((this.randomGenerator.NextDouble() < keep_probability)?1:0);
                    }

                    // arrange the mask to m examples in vectorization
                    VectorD v_rep = new VectorD();
                    for (var k = 0; k < m; k++)
                    {
                        v_rep.AddRange(v);
                        
                    }
                    this.mask.Add(v_rep);
                }
                
                float scale = 0;
                if(keep_probability > 0)
                   scale = (float)1/keep_probability;
                
                
                output = MatrixD.dot(this.mask,input) * scale;
            }


            return output;
        }
        public override MatrixD backward(MatrixD output_gradient, float learning_rate){
        
            return MatrixD.dot(output_gradient, this.mask);

            
        }

        public override Hashtable getJSONHash()
        {
            Hashtable hash = new Hashtable();
            hash["layer_type"] = "dropout";
            hash["shape_input"] = this.shape_input;
            hash["shape_output"] = this.shape_output;
            hash["mask"] = (this.mask == null)? new MatrixD():this.mask;
            hash["dropout_rate"] = this.dropout_rate;


            return hash;
        }

        public static Layer buildLayer(JsonElement element){

            Dropout layer = new Dropout((float)element.GetProperty("dropout_rate").GetDouble());
            layer.shape_input = VectorI.parseJsonElement(element.GetProperty("shape_input"));
            layer.shape_output = VectorI.parseJsonElement(element.GetProperty("shape_output"));
            layer.mask = MatrixD.parseJsonElement(element.GetProperty("mask"));


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

