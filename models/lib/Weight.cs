using System;
using System.Collections;
using System.Collections.Generic;

using types;

namespace custom_lib
{
    public enum DenseInitializationMode
    {
        xavier, kaiming_he
    }

    public class Weight
    {

        public static MatrixD getWeights(float mean, float sigma, VectorI shape, int previous_neurons, DenseInitializationMode mode){

            MatrixD W = new MatrixD();
            switch (mode)
            {
                case DenseInitializationMode.xavier:
                    // xavier: good for tanh, sigmoid and just a little with Relu
                    W = MathUtil.getMatrixRandomNormal(shape[0], shape[1], mean, sigma) * (double)Math.Sqrt((float)1/previous_neurons);
                    break;
                case DenseInitializationMode.kaiming_he:
                    // also called He, Is Kaiming He
                    // kaiming good with ReluNonlinearities. ReLU changes the activations and the variance is halved, so we need to double the variance to get the original effect of Xavier
                    // (1+a.^2) * previous_neurons), a in RelU is 0
                    int a = 0;  // other relu should change it
                    W = MathUtil.getMatrixRandomNormal(shape[0], shape[1], mean, sigma) * (double)Math.Sqrt((float)2/((1+ (a * a)) * previous_neurons));
                    break;
                default:
                    W = MathUtil.getMatrixRandomNormal(shape[0], shape[1], mean, sigma) * (double)Math.Sqrt((float)1/previous_neurons);
                    break;
            }

            return W;
            
        }

/*
else
    % default xavier: good for tanh, sigmoid and a little with Relu
    W = normrnd(mean, sigma, shape) * sqrt(1/previous_neurons);
 
end



end
*/

    }
}
