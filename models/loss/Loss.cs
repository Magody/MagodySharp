using System;

using types;

namespace loss
{
    public class Loss
    {
        private static double eps = 1e-10;
        // private static double huber_k = 1;

        // todo: optimize, use MatrixD instead VectorD?

        public static double mse(VectorD y_true, VectorD y_pred){
            VectorD diff = y_true - y_pred;
            return VectorD.mean(VectorD.dot(diff,diff));
        }

        public static VectorD mseDerivative(VectorD y_true, VectorD y_pred){
            return ((y_pred - y_true) * 2) / y_true.Count;
        }

        public static double mse(MatrixD y_true, MatrixD y_pred){
            MatrixD diff = y_true - y_pred;
            return VectorD.mean(MatrixD.dot(diff,diff).mean(1)[0]);
        }

        public static MatrixD mseDerivative(MatrixD y_true, MatrixD y_pred){
            return ((y_pred - y_true) * 2) / y_true.Count;
        }


        public static double logcosh(VectorD y_true, VectorD y_pred){
            VectorD diff = y_true - y_pred;

            return VectorD.sum(VectorD.log(VectorD.cosh(diff)));
        }

        public static VectorD logcoshDerivative(VectorD y_true, VectorD y_pred){
            VectorD diff = y_true - y_pred;
            return VectorD.tanh(diff);
        }
        
        public static double binaryCrossEntropy(VectorD y_true, VectorD y_pred){
            // loss = mean(-y_true .* log(y_pred+Loss.eps) - (1 - y_true) .* log(1 - y_pred + Loss.eps));

            int n = y_true.Count;

            return  VectorD.mean(
                        VectorD.dot(
                            VectorD.dot(
                                (y_true * -1),
                                VectorD.log(y_pred + Loss.eps)
                            ) - (VectorD.ones(n) - y_true),
                            VectorD.log((VectorD.ones(n) - y_pred) + Loss.eps)
                        )
                    );
        }

        public static double binaryCrossEntropy(MatrixD y_true, MatrixD y_pred){
            // loss = mean(-y_true .* log(y_pred+Loss.eps) - (1 - y_true) .* log(1 - y_pred + Loss.eps));

            int n = y_true.Count;
            int m = y_true[0].Count;

            return  VectorD.mean(
                        MatrixD.dot(
                            MatrixD.dot(
                                (y_true * -1),
                                MatrixD.log(y_pred + Loss.eps)
                            ) - (MatrixD.ones(n,m) - y_true),
                            MatrixD.log((MatrixD.ones(n,m) - y_pred) + Loss.eps)
                        ).mean(1)[0]
                    );
        }

        public static VectorD binaryCrossEntropyDerivative(VectorD y_true, VectorD y_pred){
            // loss = ((1 - y_true) ./ (1 - y_pred) - y_true ./ y_pred) / length(y_true);
            int n = y_true.Count;
            return (
                VectorD.dotDivide(
                    VectorD.ones(n) - y_true,
                    VectorD.ones(n) - y_pred
                ) - VectorD.dotDivide(y_true, y_pred)
            ) / n;
        }    

        public static MatrixD binaryCrossEntropyDerivative(MatrixD y_true, MatrixD y_pred){
            // loss = ((1 - y_true) ./ (1 - y_pred) - y_true ./ y_pred) / length(y_true);
            int n = y_true.Count;
            int m = y_true[0].Count;
            return (
                MatrixD.dotDivide(
                    MatrixD.ones(n,m) - y_true,
                    MatrixD.ones(n,m) - y_pred
                ) - MatrixD.dotDivide(y_true, y_pred)
            ) / n;
        }        

        // special
        public static VectorD softmaxGradient(VectorD y_true, VectorD y_pred){
            // loss = y_pred - y_true;
            return y_pred - y_true;
        }

        public static MatrixD softmaxGradient(MatrixD y_true, MatrixD y_pred){
            // loss = y_pred - y_true;
            return y_pred - y_true;
        }


        /*
        Pending to pass from Octave code:          

            % huber
            function loss = huber(y_true, y_pred)
                delta = Loss.huber_k;
                if nargin == 3
                    delta = k;
                end
                
                residual = y_true - y_pred;
                residual_abs = abs(residual);
                
                value_residual_less_equal = 0.5 * residual .^ 2;
                value_residual_greater = delta * residual_abs + 0.5 * delta * delta;
                
                less_equal_than_delta = residual_abs <= delta;
                
                loss_result = zeros(size(less_equal_than_delta));
                
                for i=1:length(less_equal_than_delta)
                    if less_equal_than_delta(i)
                        loss_result(i) = value_residual_less_equal(i);
                    else
                        loss_result(i) = value_residual_greater(i);
                    end
                end
                
                loss = mean(loss_result);
            end
            
            function loss = huber_derivative(y_true, y_pred)
                
                loss = sign((y_true - y_pred))/2;
                %{
                delta = Loss.huber_k;
                if nargin == 3
                    delta = k;
                end
                
                residual = y_true - y_pred;
                
                loss = zeros(size(residual));
                
                % clip function, also can be done with signum function
                for i=1:length(residual)
                    r = residual(i);
                    if r < -delta
                        loss(i) = -delta;
                    elseif -delta <= r && r <= delta
                        loss(i) = r;
                    elseif r > delta
                        loss(i) = delta;
                    end
                end
                %}
                    
            end
        
        */


    }
}
