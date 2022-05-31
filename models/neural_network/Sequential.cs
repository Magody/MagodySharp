using System;
using System.Collections.Generic;
using System.Collections;
using layers;
using types;


using System.Text.Json;


namespace neural_network
{

    public class Sequential
    {

        public List<Layer> network {get; set;}
        public VectorI shape_input {get; set;}
        public VectorI shape_output {get; set;}

        public Sequential(){
            
        }

        public Sequential(List<Layer> network){
            this.network = network;

            int n = network.Count;

            if(n == 0){
                this.shape_input = new VectorI{0,0};
                this.shape_output = new VectorI{0,0};
            }
            else{
                Layer first_layer = network[0];

                this.shape_input = first_layer.shape_input;
                this.shape_output = first_layer.shape_output;

                for (var index_layer = 1; index_layer < n; index_layer++)
                {
                    this.shape_output = network[index_layer].init(this.shape_output);
                }
            }
                
            
        }

        public Sequential clone(){
            List<Layer> network_copy = new List<Layer>();
            foreach (var layer in this.network)
            {
                network_copy.Add(layer.clone());
            }

            return new Sequential(network_copy);

        }

        public MatrixD forward(MatrixD x, Hashtable context){
            
            int n = this.network.Count;

            MatrixD output = x;
            for (var index_layer = 0; index_layer < n; index_layer++)
            {
                output = this.network[index_layer].forward(output, context);
            }
            
            return output;
        }

        public MatrixD backward(MatrixD initial_gradient, float alpha){
            
            int n = this.network.Count;

            MatrixD grad = initial_gradient;


            for (var index_layer = n-1; index_layer >= 0; index_layer--)
            {
                grad = this.network[index_layer].backward(grad, alpha);
            }
            
            return grad;
        }

        public Hashtable getJSONHash(){
            Hashtable hash = new Hashtable();

            List<Hashtable> list_json_network = new List<Hashtable>();

            foreach (var layer in this.network)
            {
                list_json_network.Add(layer.getJSONHash());
            }

            hash["network"] = list_json_network;
            hash["shape_input"] = this.shape_input;
            hash["shape_output"] = this.shape_output;

            return hash;
        }

        // layer_type

        public static Sequential buildSequential(JsonElement element){

            JsonElement net;
            bool is_valid = element.TryGetProperty("network", out net);

            if(!is_valid){
                return null;
            }

            List<JsonElement> network_json = JsonSerializer.Deserialize<List<JsonElement>>(element.GetProperty("network").GetRawText());

            List<Layer> network = new List<Layer>();

            foreach (JsonElement item in network_json)
            {
                string layer_type = item.GetProperty("layer_type").GetString();

                switch (layer_type)
                {
                    case "dense":
                        network.Add(Dense.buildLayer(item));
                        break;
                    case "activation":
                        network.Add(Activation.buildLayer(item));
                        break;
                    case "dropout":
                        network.Add(Dropout.buildLayer(item));
                        break;
                    default:
                        throw new Exception("Layer unknown " + layer_type);
                }
                
            }

            Sequential sequential = new Sequential(network);
            sequential.shape_input = VectorI.parseJsonElement(element.GetProperty("shape_input"));
            sequential.shape_output = VectorI.parseJsonElement(element.GetProperty("shape_output"));
            

            return sequential;
        }

        public override string ToString(){

            string output = "";

            foreach (Layer layer in this.network)
            {
                output += $"{layer.ToString()}\n";
            }

            return output;
        }

        
        
    }
}
