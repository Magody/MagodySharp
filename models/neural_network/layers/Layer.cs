using System;

using System.Collections;
using System.Collections.Generic;

using types;

namespace layers
{
    
    public abstract class Layer
    {
        public abstract VectorI shape_input {get; set;}
        public abstract VectorI shape_output {get; set;}

        // init shapes and others
        public abstract VectorI init(VectorI shape_input);
        public abstract MatrixD forward(MatrixD input, Hashtable context);
    
        // update parameters and return input gradient
        public abstract MatrixD backward(MatrixD output_gradient, float learning_rate);

        public abstract Layer clone();

        public abstract Hashtable getJSONHash();
        

    }
}